# Experiment registry

One section per experiment. Each states the question, the hypothesis being
risked, the oracle that could contradict it, the protocol in enough detail to
rerun, and an empty **Result** block to fill in after the run.

The hypothesis is written down *before* the run. A hypothesis recorded after
seeing the numbers is not a hypothesis, and the point of writing them here is
to make it visible when one fails.

Common conventions: `--quick` runs against the tiny locally-constructed model
with the network disabled and writes `<name>_quick.json`; `--seed` defaults to
`0`; every JSON carries the metadata block from SPEC §8, including the budget
actually spent.

Default models, stated once: the **cheap tier** is a 0.5B instruction-tuned
model, the **main tier** a 1.5B instruction-tuned model specialised for
mathematics. Datasets: grade-school word problems for the main corpus, a
competition-mathematics subset for the harder split, and a program-synthesis
set for the code verifier. Exact ids and revisions go in each results JSON,
not here, because they change and a stale id in a document is worse than no id.

---

## F1 — `tree_engine.py`

**Question.** What does sharing one KV cache across a tree of reasoning traces
buy, and does it change the logits?

**Hypothesis.** Speedup against independent re-prefill grows roughly linearly
in depth at fixed branching factor, because the wasted work in the baseline is
the repeated prefix. Peak memory is lower by roughly the prefix-sharing
fraction. Logit deviation is at the level of floating-point reassociation —
around `1e-5` — and if it is larger than `1e-3` something is wrong rather than
merely imprecise.

**Oracle.** Independent re-prefill of each path from the root: the same
computation done the slow, obviously-correct way.

**Protocol.**

* Branching factor `k in {2, 4, 8}` x depth `in {2, 4, 8, 16}`, each with 16
  trees, on both the cheap-tier model and the tiny test model.
* Measure: wall time, peak allocated memory, blocks allocated, block waste at
  branch points, and the maximum absolute and relative logit deviation against
  re-prefill at every node.
* Separately: eviction under a cache budget deliberately set to 50% of what
  the tree needs, confirming exactness after recomputation and reporting the
  recomputed-token count.
* Also validate the analytic FLOP model: compare predicted FLOPs against
  measured throughput on a fixed workload and report the ratio. Every FLOP
  axis in the repository inherits this number.
* If the optional fast inference engine is installed, record its throughput on
  the same workload as a reference point — not as a baseline the project
  claims to beat, but so the reader knows the absolute scale.

**Outputs.** `results/tree_engine.{json,png}`. Figure: speedup against depth,
one line per branching factor; second panel of peak memory against tree size
for shared and independent; third panel of logit deviation against depth, on a
log axis, which should be flat if the implementation is right and rising if
positions are drifting.

**Pass condition.** Max absolute logit deviation under `1e-3` at every node,
and exactness preserved across an eviction-and-recompute cycle.

**Compute.** Minutes on GPU; the tiny-model arm runs on CPU in CI.

**Result.** *(fill in: speedup table, memory, deviation, the FLOP-model ratio)*

---

## F2 — `value_ground_truth.py`

**Question.** Can the value of a reasoning prefix be pinned down to a useful
interval at an affordable cost, and does the adaptive allocator's interval
mean what it says?

**Hypothesis.** Adaptive allocation reaches a target half-width of 0.10 for
roughly 35-45% fewer tokens than a fixed `N = 32`, because variance is
`p(1-p)/n` and a large fraction of prefixes sit near 0 or 1. The adaptive
procedure's realised coverage will be **below** its nominal 95%, because
stopping on a narrow interval is optional stopping; the shortfall is the
number this experiment exists to produce.

**Oracle.** For the coverage claim, simulation against known `p` — the only
case where the truth is available by construction. For the corpus itself, the
Wilson interval is the honest statement of what is known.

**Protocol.**

* Coverage simulation first, before any generation: true `p` on a 21-point
  grid over `[0, 1]`, `n` in `{4, 8, 16, 32, 64}`, 20000 replicates each, for
  fixed-`n` and for the adaptive rule. Report realised coverage and the
  widening factor that restores nominal coverage.
* Then the corpus: 500 problems, one sampled trace each, prefixes at every
  step boundary including the empty prefix, `n_min = 4`, `n_max = 64`, target
  half-width `0.10`, temperature stated and fixed.
* Record the full allocation histogram: how many prefixes settled at each `n`.
* Splits by problem, 60/20/20, with disjointness asserted.
* Report the token cost against the fixed-`N = 32` counterfactual, computed
  from the same realised completion lengths rather than from an assumed mean.

**Budget arithmetic**, to be replaced by measured values:
500 problems x ~8 prefixes x ~20 mean rollouts x ~150 tokens is about 12M
generated tokens. A fixed `N = 32` over the same prefixes is about 19M. At
2500-4000 tokens/second on the fast path that is roughly an hour; on the
project's own batched decode, closer to six.

**Outputs.** `results/value_ground_truth.{json,png}`. Figure: realised against
nominal coverage for both allocation rules; the allocation histogram; the
distribution of half-widths achieved; cost against fixed-`N`.

**Result.** *(fill in: the coverage shortfall and widening factor, corpus
size, median half-width, tokens spent, saving)*

---

## F3 — `verifier_audit.py`

**Question.** How often is the answer checker wrong, and in which direction?

**Hypothesis.** False rejects outnumber false accepts, concentrated on
non-numeric answers — intervals, sets, expressions, answers with units — and
the symbolic stage's timeout accounts for a measurable share. False accepts
will be rarer and will come from the last-number extraction rule firing on a
trace that never stated an answer.

**Oracle.** Hand labels, 200 samples, stratified 100 accepted / 100 rejected,
sampled at random within each stratum.

**Protocol.**

* Sample from the traces generated in F2, so the audit is of the distribution
  the project actually verifies rather than of a convenient one.
* Label by hand. Record the labelling procedure and any case where the correct
  label was genuinely unclear, as a separate count — those exist and hiding
  them in one of the two buckets misstates the rate.
* Report false-accept and false-reject rates with Wilson intervals, broken
  down by answer type and by which extraction rule fired.
* Repeat for the code verifier: a test-passing program that is wrong, and a
  correct program the harness fails, are both findable by inspecting
  disagreements between the sandbox result and a hand reading.

**Outputs.** `results/verifier_audit.{json,png}`. Figure: rates by answer
type with intervals; extraction-rule breakdown.

**Pass condition.** None; it measures. But the resulting false-accept rate is
the floor quoted beside every calibration number in F4, and if it exceeds 5%
the write-up must discuss what that does to the rest of the project rather
than proceeding as if it did not.

**Compute.** Generation already done in F2; the cost is the hand labelling.

**Result.** *(fill in)*

---

## F4 — `prm_calibration.py`

**Question.** How well does a process reward model predict the value of a
prefix, and does the training objective matter?

**Hypothesis.** The binomial objective beats MSE on the point estimate by a
small but real margin, concentrated on prefixes with few rollouts. The
hard-label objective is clearly worse and, more importantly, *miscalibrated
upward* by close to the analytic `1-(1-p)^n` prediction. Self-consistency
agreement is a surprisingly strong confidence signal and will be competitive
with the PRM on calibration while being worse on resolution.

**Oracle.** Held-out `V*` from F2's corpus, with its intervals widened by
F2's measured coverage factor.

**Protocol.**

* Train: ORM; PRM with each of the three objectives; both a frozen base with
  a trained head and a base with low-rank adapters. Three seeds each; report
  spread, because single-seed comparisons at this scale are noise.
* Evaluate on the held-out problem split: ECE with equal-mass bins and a
  bootstrap interval over problems, the reliability diagram, the full Murphy
  decomposition, Spearman against `V*`, and AUC for separating eventual
  success from failure.
* Measure the hard-label bias directly: for prefixes where `V*` is well
  determined, compare the hard-label target `1[k>0]` against `V*` and against
  the analytic `1-(1-V*)^n`, and report the agreement with the prediction.
* Quote the verifier's false-accept rate from F3 next to every ECE.

**Outputs.** `results/prm_calibration.{json,png}`. Figure: reliability
diagrams, one panel per objective, with bootstrap bands; a second figure of
the Murphy components as stacked bars; a third of measured against predicted
hard-label bias.

**Result.** *(fill in the README table, plus which of the four hypothesis
clauses survived)*

---

## F5 — `step_localization.py`

**Question.** Does a process reward model identify the step where a trace went
wrong, better than a degenerate baseline?

**Hypothesis.** Exact localisation will be modest — well under half — and
within-one-step considerably better. The "guess the last step" baseline will
be stronger than expected, because errors in generated traces are not
uniformly distributed over step indices. The margin over the best degenerate
baseline is the result, and it may be small.

**Oracle.** The known injection index, filtered by a measured `V*` drop.

**Protocol.**

* Take correct traces from F2. Inject exactly one error at a uniformly sampled
  step index, from six types: arithmetic slip, dropped constraint, wrong
  formula, sign error, off-by-one, and a valid-but-non-advancing step.
* Filter: keep an injection only if `V*` measured after the injected step is
  lower than `V*` before it by more than the combined interval width. Report
  the discard rate **per error type** — some types will be routed around much
  more often than others, and that is itself informative about what the model
  can recover from.
* Metrics: exact-hit rate, within-`k` for `k in {1, 2}`, and per-step
  precision/recall at a swept threshold.
* Baselines, all three: always the last step, always the longest step, uniform
  random. Also a "score drop at the true index" oracle upper bound, computed
  from `V*` itself, which bounds what any scorer of this kind could achieve.

**Outputs.** `results/step_localization.{json,png}`. Figure: hit rate by error
type with the baselines as horizontal lines; discard rate by error type; the
precision/recall curve against the `V*` oracle bound.

**Result.** *(fill in, with the margin over the best baseline stated plainly
even if it is small)*

---

## F6 — `matched_compute.py`

**Question.** At equal compute, which way of spending it works best?

**Hypothesis.** (a) Self-consistency is the strongest method per FLOP at small
budgets and saturates; (b) best-of-N with a PRM overtakes it at larger budgets
but by less than the samples-based comparisons in the literature suggest, once
the PRM's own forward cost is counted; (c) tree search beats best-of-N on the
harder split and not on the easier one; (d) the ordering changes between the
token axis and the FLOP axis for at least one pair of methods; (e) PRM-min
beats PRM-last as an aggregation; (f) difficulty-aware allocation gives a real
but modest gain, far smaller than the version computed with test-set leakage.

**Oracle.** The FLOP accounting, validated in F1; pass@k as the ceiling; and
paired bootstrap intervals, which can and should contradict an apparent
ordering.

**Protocol.**

* 200 problems from the main split, 200 from the harder split, reported
  separately — pooling them would let the easy split carry the conclusion.
* Budgets: 1x, 2x, 4x, 8x, 16x, 32x, 64x the token cost of a single sample.
* Methods: single sample, self-consistency, best-of-N with each of the four
  PRM aggregations and with the ORM, step beam, lookahead beam, MCTS.
* Every method receives a token budget, not a sample count, and reports its
  realised budget; a method that overspends by more than 5% is a failed run,
  not a data point.
* The sequential-against-parallel arm: at each budget, compare `n` samples of
  length `L` against `n/2` samples with twice the token limit.
* Paired bootstrap over problems, 1000 resamples; report every comparison as a
  difference with an interval, and mark overlapping pairs as not
  distinguishable.

**Budget arithmetic.** 400 problems x 127x single-sample budget x ~200 tokens
is about 10M tokens per method; across nine method arms, roughly 90M generated
tokens plus reward-model forwards. This is the most expensive experiment in
the project. Reduce by dropping to 150 problems per split before dropping
budgets or methods, and record the reduction in `scope`.

**Outputs.** `results/matched_compute.{json,png}`. Figure: accuracy against
FLOPs, one line per method, with bootstrap bands, one panel per split; a second
panel on the token axis so the ordering change is visible; a third of the
sequential-against-parallel comparison; a fourth of allocation strategies.

**Result.** *(fill in the README table, plus which of (a)-(f) survived)*

---

## F7 — `grpo_training.py`

**Question.** Does reinforcement learning against a verifiable reward add
capability, or redistribute probability mass the model already had?

**Hypothesis.** pass@1 improves clearly. pass@64 does not improve, and may
decline. Generation entropy falls throughout and the number of distinct correct
solutions per solved problem falls with it. The normalised advantage is less
stable than the centred one on the easy split, where many groups have zero
variance. Length grows under sequence-level normalisation and much less under
token-level.

**Oracle.** pass@k at matched sampling temperature, which is the direct test of
whether the support changed; and a contamination-audited held-out split, so a
gain cannot be memorisation surfacing.

**Protocol.**

* Base: the main-tier model with low-rank adapters, rank and targets recorded.
* 500 training prompts from the training split only, `G = 8`, roughly 600
  update steps, verifiable binary reward.
* Arms: centred against normalised advantage; token-level against
  sequence-level normalisation; `beta = 0` against a stated KL coefficient.
  Each arm two seeds; this is enough to see whether a difference is plausible
  and not enough to establish a small one, and the write-up must say that.
* Evaluate before and after on 200 held-out problems: pass@1, pass@8, pass@64,
  all at the same temperature, with distinct-solution counts and entropy.
* Log every diagnostic in SPEC §6.3 every step.

**Budget arithmetic.** Rollouts: 600 steps x 8 samples x ~250 tokens is about
1.2M generated tokens per arm. Evaluation: 200 problems x 64 samples x ~200
tokens x 2 (before and after) is about 5M tokens, and it dominates. Six arms
is therefore mostly evaluation cost; share the "before" evaluation across arms
rather than recomputing it, and say so.

**Outputs.** `results/grpo_training.{json,png}`. Figure: reward, entropy,
length and distinct-solutions against step, one panel each, all arms; a second
figure of pass@k before and after as a function of `k`, which is the one that
answers the question.

**Result.** *(fill in, including the direct answer: sharpening or capability)*

---

## F8 — `selective_prediction.py`

**Question.** If the model may abstain, how much accuracy can it buy with how
much coverage, and which confidence signal is best?

**Hypothesis.** Self-consistency agreement beats the PRM score at high
coverage and loses at low coverage, because agreement saturates at 1 and
cannot rank the easy cases against each other. Mean token log-probability is
the weakest signal and is included because it is the cheapest.

**Oracle.** The realised error rate at each coverage level, with bootstrap
intervals.

**Protocol.**

* 400 problems across both splits, 16 samples each.
* Signals: PRM score under each aggregation, self-consistency agreement
  fraction, mean token log-probability, predictive entropy, and the ORM score.
* Report the risk-coverage curve, AURC with a bootstrap interval, and accuracy
  at 50% and 80% coverage.

**Outputs.** `results/selective_prediction.{json,png}`. Figure: risk-coverage
curves with bands; AURC bar chart with intervals.

**Result.** *(fill in)*

---

## F9 — `contamination.py`

**Question.** Are the evaluation problems in anything the model or the reward
models were trained on?

**Hypothesis.** No overlap with this project's own training data, by
construction. The completion probe is the uncertain one and is the reason the
experiment exists: a pretrained model may already have seen these problems,
and no overlap check against this project's data would reveal it.

**Oracle.** A planted duplicate, to confirm the overlap check can find what it
claims to; and for the probe, the model's own verbatim reproduction rate on
text it certainly has not seen, as a null.

**Protocol.**

* 13-gram overlap between every evaluation problem and: the training split,
  the `V*` corpus, and the reinforcement-learning prompt set. Report the
  fraction of evaluation problems with any match and the match-length
  distribution.
* Completion probe: give the first half of each problem statement, sample
  greedily, and measure normalised edit distance to the true second half.
  Compare against a null of freshly-written problems in the same style, which
  the model cannot have memorised. The null is what makes the probe
  interpretable.
* Report the numbers whatever they show, and if the probe suggests
  contamination, re-report F6's and F7's headline numbers on the
  lowest-suspicion subset alongside the full set.

**Outputs.** `results/contamination.{json,png}`.

**Result.** *(fill in)*

---

## Where the measurements disagreed

Collect here, as they are found, the places where a measurement contradicted
the expected answer — including expectations taken from the literature. Each
entry: what was expected, what was measured, and what was done about it.

This section being empty at the end of phase 10 means the experiments were not
sharp enough, not that everything worked.
