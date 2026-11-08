# arborist — technical specification

This document is the contract. Where it names a function signature, a formula
or a metric, implement exactly that; where it leaves a choice open it says so.
Revisions go at the bottom.

---

## 0. Package layout and public surface

```
src/arborist/
  __init__.py            re-exports the flat public API
  engine/
    policy.py            Policy: a batched sampling wrapper over a causal LM
    segment.py           step segmentation and its failure accounting
    kvtree.py            TraceTree, BlockCache, copy-on-write branching
    budget.py            Budget: the single source of truth for compute spent
  verify/
    answer.py            extraction and equivalence for maths answers
    sandbox.py           resource-limited execution of generated code
    audit.py             false-accept / false-reject measurement
  value/
    wilson.py            binomial intervals, and their simulated coverage
    montecarlo.py        V* estimation
    allocate.py          sequential rollout allocation under a budget
    dataset.py           the V* corpus: build, store, split, load
  reward/
    heads.py             value heads over a frozen or adapted base model
    losses.py            binomial, MSE-on-p-hat, and hard-label objectives
    prm.py               step-level scoring and its aggregations
    orm.py               outcome-level scoring
    calibrate.py         ECE, reliability diagrams, Murphy decomposition
  search/
    consistency.py       majority vote
    bon.py               best-of-N with the aggregation choices
    beam.py              step-level beam and lookahead beam
    mcts.py              PUCT over the trace tree
    allocator.py         difficulty-aware budget allocation
  inject/
    perturb.py           controlled single-step error injection
    bench.py             the filtered benchmark and its metrics
  rl/
    lora.py              low-rank adapters, written out
    rollout.py           group rollout collection
    grpo.py              group-relative policy optimisation
    diagnostics.py       entropy, length, diversity, KL, gradient norms
  eval/
    passk.py             the unbiased pass@k estimator
    matched.py           the matched-compute harness
    selective.py         risk-coverage and AURC
    contamination.py     n-gram overlap and the completion probe
    stats.py             bootstrap intervals and paired tests
```

`arborist.__init__` re-exports, and nothing else: `Policy`, `TraceTree`,
`Budget`, `segment_steps`, `check_answer`, `run_sandboxed`, `estimate_value`,
`ValueEstimate`, `ProcessRewardModel`, `OutcomeRewardModel`, `search`,
`pass_at_k`, `matched_compute_curve`, `train_grpo`.

---

## 1. The engine

### 1.1 `Policy`

A thin, explicit wrapper over a causal language model. It exists so that every
token generated anywhere in the repository passes through one place that counts
it.

```python
@dataclass(frozen=True)
class SamplingParams:
    temperature: float
    top_p: float = 1.0
    max_new_tokens: int = 512
    stop: tuple[str, ...] = ()
    seed: int | None = None

class Policy:
    def __init__(self, model, tokenizer, *, device, dtype, budget: Budget): ...
    def generate(self, prompts: Sequence[str], params: SamplingParams, *,
                 n: int = 1, generator=None) -> list[list[Completion]]: ...
    def logprobs(self, prompt: str, continuation: str) -> Tensor: ...
    def score_batch(self, texts: Sequence[str]) -> Tensor: ...
```

No defaults on `temperature`. The value of a prefix is defined relative to a
sampling policy (brief R2), so a defaulted temperature is a way of producing
numbers whose meaning is unrecorded.

### 1.2 Step segmentation

```python
def segment_steps(text: str, *, policy: str = "blank_line") -> SegmentResult
```

`policy` is one of `"blank_line"` (split on `\n\n`), `"numbered"` (split on a
leading `Step k:` or `k.` pattern), `"sentence"` (a stated regex), or
`"newline"`. `SegmentResult` carries the steps, their character and token
spans, and a `fallback_used` flag.

The honest part: step-level claims are claims about a chunking, and the
chunking fails sometimes. `SegmentResult` must record it, and every experiment
that segments reports the distribution of steps per trace, tokens per step,
and **the fraction of traces where the primary policy found no boundary and
the fallback was used**. A step-localisation result on a corpus where 30% of
traces are a single step means something different from one where 2% are, and
without this number the reader cannot tell which they are looking at.

### 1.3 The trace tree

A reasoning trace is a path from root to leaf; a search explores a tree; the
tree shares prefixes. Re-prefilling each branch from the root is the
straightforward implementation and wastes work quadratic in depth.

```python
@dataclass
class TraceNode:
    node_id: int
    parent: int | None
    step_text: str
    token_ids: tuple[int, ...]
    blocks: tuple[int, ...]        # cache blocks owned by this node
    depth: int
    terminal: bool
    scores: dict[str, float]       # reward-model scores, keyed by scorer name

class BlockCache:
    """Key/value tensors in fixed-size blocks, shared between siblings."""
    def __init__(self, *, n_layers, n_kv_heads, d_head, block_size=16,
                 max_blocks, device, dtype): ...
    def allocate(self, n_tokens: int) -> tuple[int, ...]: ...
    def gather(self, block_lists: Sequence[Sequence[int]]) -> tuple[Tensor, Tensor]: ...
    def release(self, blocks: Sequence[int]) -> None: ...

class TraceTree:
    def __init__(self, policy: Policy, prompt: str, *, cache: BlockCache): ...
    def expand(self, node_ids: Sequence[int], *, k: int,
               params: SamplingParams, generator=None) -> list[list[int]]: ...
    def path_text(self, node_id: int) -> str: ...
    def evict(self, keep: Collection[int]) -> int: ...
```

Design points, all of which belong in the module docstring:

* **Blocks, not per-sequence tensors.** A node's KV lives in whole blocks it
  owns; its ancestors' blocks are referenced, never copied. Branching is
  therefore free until the child writes its first token, at which point it
  allocates its own block. The last block of a parent is the awkward case: if
  it is partially filled, two children writing into it would collide, so a
  child always starts a fresh block and the parent's partial block is read-only
  from then on. This wastes up to `block_size - 1` slots per branch point and
  is the right trade; say so, and report the measured waste.
* **Gathering, not a custom kernel.** `gather` assembles contiguous K/V
  tensors for a batch of block lists and hands them to standard attention.
  This is not as fast as a paged-attention kernel and the repository does not
  claim it is. The measured claim in F1 is against independent re-prefill,
  which is the baseline the optimisation actually replaces.
* **Eviction with recomputation.** Under memory pressure, evict the leaves
  whose blocks are referenced by nothing else, least-recently-used first. A
  node keeps its `token_ids`, so an evicted subtree can be re-prefilled on
  demand; eviction is a cost, not a loss. `evict` returns the number of blocks
  reclaimed, and `Budget` records the recomputation.
* **Exactness.** `benchmarks/cross_check_cache.py` runs the same set of
  prefixes through the shared-cache path and through independent re-prefill
  and reports the maximum absolute and relative logit deviation. Pass condition
  is `max_abs < 1e-3` in float32 with the achieved number published; a
  deviation at the `1e-2` level means a block-boundary or position-id bug, and
  the script should bisect by depth to find where it starts.

Position ids are the single most likely bug in this component. A node's tokens
occupy absolute positions `[len(prefix), len(prefix) + len(tokens))`, and the
positions must be computed from the path length, never from the block index.

### 1.4 Compute accounting

```python
@dataclass
class Budget:
    prefill_tokens: int = 0
    decode_tokens: int = 0
    recompute_tokens: int = 0
    reward_tokens: int = 0
    verifier_calls: int = 0
    wall_seconds: float = 0.0

    def flops(self, cfg) -> int: ...
```

The analytic cost of one forward token at context length `T`, per layer:

```
  qkvo        2 * (d*d + 2*d*d_kv + d*d)       d_kv = n_kv_heads * d_head
  attention   2 * 2 * T * d_head * n_heads
  mlp         2 * n_mat * d * d_mlp            n_mat = 2 plain, 3 gated
```

plus `2 * d * V` once for the unembedding. Backward is twice forward. Prefill
of `p` tokens pays the attention term at a mean context of `p/2`; decoding the
`i`-th token pays it at `p + i`.

Two requirements:

* Reward-model forwards count. A process reward model scoring every step of
  every branch can cost more than the generation, and a comparison that omits
  it ranks methods wrongly. `reward_tokens` is separate so the split is
  visible.
* The analytic count is validated once against measured throughput, in F1, and
  the ratio is reported. If the analytic model is off by 2x, every FLOP axis
  in the repository is off by the same factor and the reader should be told.

---

## 2. Verification

### 2.1 Maths answers

```python
def check_answer(predicted: str, reference: str) -> AnswerCheck
```

Staged, with the stage that decided recorded in the result:

1. **Extraction** — `\boxed{...}`, then `the answer is ...`, then the last
   number in the text. Record which rule fired; the fraction of traces where
   extraction fell through to the last number is a quality signal for the
   whole corpus.
2. **Normalisation** — strip whitespace, `\left`/`\right`, trailing units,
   `\!`, `\%`, commas in numerals, `\dfrac` to `\frac`; canonicalise `0.5`
   against `1/2`.
3. **Numeric comparison** — if both parse as floats, compare with relative
   tolerance `1e-6`.
4. **Symbolic comparison** — parse both with a restricted symbolic parser and
   test `simplify(a - b) == 0` under a hard timeout. Symbolic equivalence
   checking does not always terminate; the timeout is part of the method and
   its hit rate is reported.

`AnswerCheck` carries `correct`, `stage`, `extracted`, `timed_out`.

### 2.2 The sandbox

```python
def run_sandboxed(source: str, *, stdin: str = "", cpu_seconds: float = 2.0,
                  memory_bytes: int = 512 << 20, timeout: float = 5.0,
                  n_proc: int = 0) -> SandboxResult
```

A fresh subprocess with `RLIMIT_CPU`, `RLIMIT_AS`, `RLIMIT_NOFILE`,
`RLIMIT_NPROC` and `RLIMIT_FSIZE` set before `exec`, a wall-clock timeout
enforced by the parent, an empty environment, and a scratch working directory
that is deleted afterwards.

State the limits of this honestly in the module docstring: resource limits
stop runaway programs, not deliberately hostile ones. A pure-Python sandbox
cannot prevent network access or filesystem reads outside the scratch
directory without OS-level support. The docstring must say that code from an
untrusted source belongs in a container or a VM, and that this module's
purpose is to survive generated code that loops forever or allocates 40 GB —
which is what generated code actually does.

`tests/test_sandbox.py` contains a program for each limit that tries to exceed
it and asserts it was stopped by that limit specifically, not by the wall
clock. A sandbox whose tests only check that correct code runs is untested.

### 2.3 The audit

```python
def audit_verifier(samples, labels) -> AuditReport
```

`samples` are (predicted, reference, model_output) triples; `labels` are hand
assigned. Report false-accept rate, false-reject rate, and both with Wilson
intervals, because an audit on 200 samples has an interval of several points
and quoting a bare rate overstates it.

Minimum 200 hand-labelled samples, stratified: 100 the checker accepted and
100 it rejected, sampled at random within each. Record the sampling procedure
in the report — an audit of a convenience sample is not an audit.

The false-accept rate propagates: a reward model trained on labels that are
wrong `e` of the time cannot be calibrated better than `e`, and every
calibration number in §4 is quoted with this floor alongside it.

---

## 3. The ground truth

### 3.1 The estimator

```python
@dataclass(frozen=True)
class ValueEstimate:
    successes: int
    trials: int
    point: float            # successes / trials
    lower: float            # Wilson
    upper: float
    conditions: SamplingConditions   # model id + revision, temperature,
                                     # top_p, max_new_tokens, stop, prompt
                                     # template, verifier version
```

The Wilson score interval for `k` of `n` at confidence `1 - alpha`, `z` the
normal quantile, `p = k/n`:

```
  centre     = (p + z^2/(2n)) / (1 + z^2/n)
  half_width = z/(1 + z^2/n) * sqrt( p(1-p)/n + z^2/(4 n^2) )
```

Wilson rather than the normal approximation because `V*` is frequently near 0
or 1, where the normal interval leaves the unit interval and has coverage far
below nominal — which is precisely the regime where prefixes are most
informative.

`SamplingConditions` is mandatory and the serialiser raises without it
(brief R2). `ValueEstimate.__str__` prints the interval, never the point alone.

### 3.2 `V*` and step advantage

```python
def estimate_value(policy, prefix, problem, *, n, params, verifier,
                   generator=None) -> ValueEstimate

def step_advantages(policy, trace, problem, *, n, params, verifier
                    ) -> list[AdvantageEstimate]
```

`A*(step_t) = V*(prefix_t) - V*(prefix_{t-1})`, and its interval is **not** the
difference of the two intervals. Because the two estimates are independent
binomials, use the Newcombe interval for the difference of two proportions, or
a bootstrap over the pooled rollouts; whichever is chosen, say which, and
validate its coverage by simulation (§3.4). Subtracting interval endpoints is
the common error and gives an interval roughly 40% too wide.

### 3.3 Adaptive allocation

Variance of a binomial proportion is `p(1-p)/n`, so prefixes near 0 or 1 reach
a given interval width with far fewer rollouts. A fixed `N` therefore spends
most of its budget on prefixes that were already settled.

```python
def allocate(policy, prefixes, *, target_half_width, n_min=4, n_max=64,
             total_budget_tokens=None, generator=None) -> list[ValueEstimate]
```

Sample in rounds: all prefixes get `n_min`, then each round doubles the
allocation only for prefixes whose current half-width exceeds the target,
stopping at `n_max` or when the token budget is exhausted. Report, per prefix,
the `n` actually used.

**The catch, which must be handled rather than ignored:** stopping when an
interval becomes narrow is optional stopping, and it breaks the nominal
coverage of the interval it stopped on. So the coverage of this procedure is
not assumed — it is measured.

### 3.4 Validating the instrument

`value/wilson.py` includes a simulation harness, and `tests/test_wilson.py`
runs it: for true `p` on a grid over `[0, 1]` and `n` on a grid, simulate
many draws and measure the **actual** coverage of the nominal 95% interval,
for the fixed-`n` case and for the adaptive procedure of §3.3.

Report both. If adaptive allocation's coverage is 89% where it claims 95%,
that is a real property of the design and the `V*` corpus's intervals are
widened by the measured factor before anything downstream uses them. This is
the same discipline as checking a numerical routine against an
arbitrary-precision reference: the measuring device gets measured first.

### 3.5 The corpus

```python
def build_corpus(problems, policy, *, prefixes_per_problem, params, verifier,
                 target_half_width, splits, seed) -> CorpusReport
```

For each problem: sample one or more full traces, segment them, and take
prefixes at step boundaries — including the empty prefix, whose `V*` is the
model's unaided pass rate on that problem and is the baseline every step
advantage is measured against.

Splits are **by problem**, never by prefix. Prefixes from one problem are
heavily correlated, and a prefix-level split leaks the problem's difficulty
into the test set, inflating every reward-model number. The split function
asserts disjointness of problem ids and the corpus report records it.

Stored as a directory of shards (gitignored) plus a manifest recording: model
id and revision, sampling conditions, verifier version, token totals, the
coverage correction from §3.4, and per-split problem id lists.

---

## 4. Reward models

### 4.1 Heads

```python
class ValueHead(nn.Module):           # a scalar logit per position
class ProcessRewardModel:             # scores each step boundary
class OutcomeRewardModel:             # scores the completed trace
```

Two configurations, both implemented and compared: a **frozen** base model
with only the head trained, and a base model with **low-rank adapters**
(§6.1) plus the head. The frozen variant is far cheaper and the comparison
decides whether the extra cost buys anything; running only the expensive one
would leave that unknown.

### 4.2 Losses

For a prefix with `k` successes out of `n` rollouts and head logit `z`:

```
  binomial   L = -( k*log(sigmoid(z)) + (n-k)*log(1-sigmoid(z)) ) / n
  mse        L = ( sigmoid(z) - k/n )^2
  hard       L = BCE( sigmoid(z), 1 if k > 0 else 0 )
```

The binomial objective is the correct one: it weights each label by the
confidence the rollouts actually support, so a prefix measured with `n=4`
pulls less than one measured with `n=64`. MSE on the point estimate treats
them equally.

The `hard` objective is included because it is what several published
process-reward corpora use, and it is **biased**: labelling a prefix positive
when any of `n` rollouts succeeded estimates `1 - (1 - V*)^n`, not `V*`. For
`V* = 0.1` and `n = 16` that is 0.81. The experiment in §F4 measures the
resulting bias on real data rather than only noting it here.

### 4.3 Step aggregation

A trace has many step scores and a search needs one number. The choice is
consequential and usually unreported, so all four are implemented and compared
in §F6: `min`, `last`, `mean`, and `prod`. `min` and `last` answer different
questions — "did anything go wrong" against "is the current state good" — and
which wins is an empirical matter.

### 4.4 Calibration

```python
def ece(probs, outcomes, *, bins=15, strategy="equal_mass", n_boot=1000)
def reliability_diagram(probs, outcomes, *, bins=15, strategy="equal_mass")
def brier_decomposition(probs, outcomes, *, bins=15) -> BrierParts
```

`strategy="equal_mass"` is the default and `"equal_width"` is available for
comparison only. Equal-width bins put most of the mass in one or two bins when
the predictor is confident, which is the usual case, and the resulting ECE is
dominated by near-empty bins.

Every ECE comes with a bootstrap interval over problems — over problems, not
over prefixes, because prefixes within a problem are correlated and bootstrapping
over them understates the interval.

The Murphy decomposition, with bins `k` of size `n_k`, mean forecast `f_k`,
observed frequency `o_k` and base rate `o`:

```
  reliability = (1/N) * sum_k n_k (f_k - o_k)^2        lower is better
  resolution  = (1/N) * sum_k n_k (o_k - o)^2          higher is better
  uncertainty = o (1 - o)                              a property of the data
  Brier       = reliability - resolution + uncertainty
```

Reporting reliability without resolution is the mistake this is here to
prevent: a model that predicts the base rate for everything has perfect
reliability, zero resolution, and no value.

---

## 5. Search

All methods share one signature and report through one `Budget`:

```python
def search(method, policy, problem, *, scorer, budget_tokens, params,
           generator=None, **method_kwargs) -> SearchResult
```

`SearchResult` carries the chosen answer, the tree, the per-method
diagnostics, and the `Budget` actually consumed. Methods are given a **token
budget**, not a sample count, so that the matched-compute comparison is a
property of the interface rather than something the experiment has to arrange.

| method | the knob | notes |
|---|---|---|
| `consistency` | `n` | majority vote over final answers; no scorer |
| `bon` | `n`, `aggregate` | one scorer call per trace (or per step for PRM aggregates) |
| `beam` | `width`, `k`, `depth` | expand each beam `k` ways per step, keep top `width` |
| `lookahead` | `width`, `k`, `rollout_depth`, `n_rollout` | score candidates by short rollouts rather than by the scorer alone |
| `mcts` | `n_sim`, `c_puct`, `k_expand`, `backup` | PUCT, below |

**PUCT.** Selection maximises

```
  Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

with the prior `P(s,a)` taken as the softmax over sampled children of their
mean token log-probability divided by a stated `tau_prior`, renormalised over
the `k_expand` children actually sampled — and the docstring must say that
this is a *sampled* prior, not the policy's true distribution over steps,
which is not available. Value backup is the mean by default with `max` as an
ablation; terminal leaves take the verifier's result, non-terminal leaves take
the scorer's. Report the tree size, the maximum depth reached and the
simulation count alongside the answer.

**Difficulty-aware allocation.**

```python
def allocate_budget(problems, *, total_budget, pilot_n, difficulty_model,
                    generator=None) -> dict[str, int]
```

Spend a small pilot per problem, estimate difficulty, then distribute the
remaining budget. The difficulty model is fitted on a **calibration split**
and applied to the test split. Using test outcomes to estimate test difficulty
is leakage, it produces a large and entirely spurious gain, and it is the
single easiest way to get this experiment wrong — so `allocate_budget`
asserts that the problem ids it is fitting on and the ones it is allocating
for are disjoint.

---

## 6. Reinforcement learning

### 6.1 Adapters

```python
class LoRALinear(nn.Module):
    """y = W x + (alpha / r) * B (A x), with W frozen."""
```

Written out rather than imported: it is forty lines, the project's house style
is to build the thing, and the scaling convention `alpha / r` is exactly the
detail that differs between implementations and silently changes the effective
learning rate. `attach_lora(model, targets, r, alpha, dropout)` returns the
trainable parameter list and asserts the base parameters are frozen.

### 6.2 Group-relative policy optimisation

For each prompt `q`, sample a group of `G` completions with rewards `r_i` from
the verifier. The advantage is group-relative, which removes the need for a
learned value function:

```
  A_i = (r_i - mean(r)) / std(r)        "normalised"
  A_i =  r_i - mean(r)                  "centred"
```

Both are implemented and §F7 reports both, because dividing by the group
standard deviation is unstable exactly where it matters: on a prompt where
every sample gets the same reward, `std(r)` is zero and the gradient is
either dropped or explodes depending on the epsilon, and prompts of that kind
are a large fraction of an easy dataset.

The token-level surrogate, with `rho_it` the importance ratio against the
behaviour policy:

```
  J = (1 / sum_i |o_i|) * sum_i sum_t min( rho_it * A_i,
                            clip(rho_it, 1-eps, 1+eps) * A_i )  -  beta * KL
```

The normaliser is the choice to be careful about. `1 / sum_i |o_i|`
(token-level) weights every token equally. `(1/G) * sum_i (1/|o_i|)`
(sequence-level) weights every *sequence* equally and thereby gives tokens in
short sequences more weight, which introduces a length bias whose direction
depends on the sign of the advantage. Implement both, default to token-level,
and measure the difference in §F7 rather than citing it.

KL to the reference policy uses the low-variance positive estimator

```
  KL_hat = ratio - log(ratio) - 1,      ratio = pi_ref / pi_theta
```

which is non-negative per sample, unlike `-log(ratio)`. `beta = 0` is a
supported configuration and is reported as an ablation, since the KL term's
value is contested.

### 6.3 Diagnostics

Logged every step, and all of them plotted in §F7 because the failure modes
are visible in these and not in the reward curve: mean reward, reward
variance, fraction of groups with zero variance, mean and maximum response
length, generation entropy, number of distinct correct answers per solved
problem, KL from the reference, gradient norm, and the clipped-token fraction.

Entropy collapse and length inflation are the two standard failure modes.
Reward going up while entropy goes to zero is the signature of the result that
§F7 exists to test.

---

## 7. Evaluation

### 7.1 pass@k

The unbiased estimator from `n >= k` samples with `c` correct:

```
  pass@k = 1 - C(n-c, k) / C(n, k)
```

computed as `1 - prod_{i = n-c+1}^{n} (1 - k/i)` for numerical stability.
Implementing it as "did any of the first `k` samples succeed" is biased and
much noisier; a test asserts agreement with the closed form on exhaustively
enumerable small cases.

pass@k is the control for the sharpening question: training that raises pass@1
while leaving pass@64 flat has redistributed mass, not added capability. Both
must be measured at the **same** sampling temperature, since lowering
temperature raises pass@1 and lowers pass@k by itself.

### 7.2 Matched compute

```python
def matched_compute_curve(methods, problems, policy, *, budgets, scorer,
                          params, n_boot=1000, seed) -> CurveReport
```

For each method and each budget: run, record accuracy, record the `Budget`
consumed. The report's x-axis is available as generated tokens and as FLOPs;
both are plotted, because the orderings differ whenever one method uses a
reward model more heavily than another.

Accuracy differences carry bootstrap intervals over problems, and because all
methods run on the same problems, a paired bootstrap on the per-problem
differences is the right test and is what `eval/stats.py` provides. Methods
whose intervals overlap are reported as not distinguishable rather than ranked.

### 7.3 Selective prediction

```python
def risk_coverage(confidences, outcomes) -> RiskCoverage
```

Sort by confidence, sweep the abstention threshold, report error against
coverage and the area under the risk-coverage curve. Confidence signals to
compare: the PRM score under each aggregation, self-consistency agreement
fraction, mean token log-probability, and predictive entropy.

### 7.4 Contamination

```python
def ngram_overlap(eval_texts, train_texts, *, n=13) -> OverlapReport
def completion_probe(policy, problems, *, prefix_fraction=0.5) -> ProbeReport
```

13-gram overlap between every evaluation problem and everything used for
training, reported as the fraction of evaluation problems with any match and
the distribution of match lengths. The completion probe gives the model the
first half of a problem statement and measures how often it reproduces the
rest verbatim, which catches contamination that reached the model before this
project did and that no overlap check against *this* project's data could find.

Report the numbers whatever they are. An audit that is only published when it
comes back clean is not an audit.

---

## 8. Cross-cutting requirements

**Determinism.** Sampling takes an explicit generator; so do the bootstrap,
the rollout allocator and every shuffle. `tests/test_determinism.py` runs each
experiment's `--quick` path twice and asserts equal JSON after dropping
`metadata.wall_seconds`. On GPU, assert `allclose` and say so.

**Tiny models for tests.** No test downloads anything. `tests/conftest.py`
constructs a randomly-initialised 2-layer, 2-head, 32-dimensional causal LM
with a 64-token vocabulary and a trivial tokenizer, which is enough to exercise
the cache, the tree, the budget, the search methods and the training step.
Tests that need real weights are marked `network` and excluded from CI.

**Results metadata.** Every `results/*.json` carries:

```json
{"metadata": {"seed": 0, "git_sha": "...", "device": "cuda:0",
              "model": {"id": "...", "revision": "..."},
              "sampling": {"temperature": 0.8, "top_p": 1.0,
                           "max_new_tokens": 512},
              "verifier_version": "...",
              "budget": {"prefill_tokens": 0, "decode_tokens": 0,
                         "reward_tokens": 0, "verifier_calls": 0,
                         "flops": 0},
              "wall_seconds": 0.0, "scope": "full",
              "command": "python experiments/....py"}}
```

`scope` is `"full"` or a sentence saying exactly how it was reduced.

---

## 9. Revisions

Append dated entries here when the specification is found to be wrong, with
what was changed and why. An empty section means nothing has been found yet,
not that nothing is wrong.
