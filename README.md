# arborist — test-time compute, measured against a ground truth that exists

Spend more compute at inference and a language model answers harder questions:
sample many reasoning traces, score them with a verifier or a reward model,
search over the steps. The literature on this is large and the claims are
mostly true. They are also mostly unmeasurable, for two reasons.

The first is that the comparisons are not at matched compute. Best-of-64 is
compared against greedy decoding, beam search against best-of-N at the same
*number of samples* rather than the same number of FLOPs, and the cost of the
reward model's own forward passes — which can exceed the cost of the
generation it is scoring — is usually left out of the accounting entirely.

The second is sharper. A process reward model claims to say *where* a
reasoning trace went wrong. That is a claim about the value of a partial
trace, and nobody checks it, because the value of a partial trace is taken to
be unknowable.

It is not unknowable. In a domain where answers can be verified, the value of
a prefix has a definition and an estimator:

```
  V*(prefix) = P( the final answer is correct | prefix, policy pi, temperature T )
```

Sample `N` completions from the prefix, verify each, and you have a binomial
proportion with a Wilson interval around it. That is a real ground truth with
quantified uncertainty, and the ground-truth credit of a single step follows
from it:

```
  A*(step_t) = V*(prefix through step t) - V*(prefix through step t-1)
```

This project builds that ground truth and then uses it to measure the things
the field normally asserts.

An important caveat, stated here because it bounds everything below: `V*` is
defined *relative to a policy and a temperature*. It is not "is this step
correct in the abstract" — it is "how often does this model, sampling this
way, get to the right answer from here". Every `V*` number in this repository
carries its model, temperature, top-p, token limit and `N`. Quoting one
without them would be meaningless, and the most common way this measurement
gets misused.

---

## 1. How well does a process reward model actually assign credit?

Trained against `V*` labels, evaluated against held-out `V*` with its interval
widths known. Calibration is reported with the Murphy decomposition, because
a well-calibrated predictor with no resolution is a constant and is useless,
and reporting only the calibration error hides that.

| | ORM | PRM (binomial loss) | PRM (MSE on p-hat) | PRM (hard labels) | self-consistency |
|---|---|---|---|---|---|
| ECE, equal-mass bins, 95% CI | `[[N1a]]` | `[[N1b]]` | `[[N1c]]` | `[[N1d]]` | `[[N1e]]` |
| Brier — reliability | `[[N1f]]` | `[[N1g]]` | `[[N1h]]` | `[[N1i]]` | `[[N1j]]` |
| Brier — resolution | `[[N1k]]` | `[[N1l]]` | `[[N1m]]` | `[[N1n]]` | `[[N1o]]` |
| Spearman against `V*` | `[[N1p]]` | `[[N1q]]` | `[[N1r]]` | `[[N1s]]` | `[[N1t]]` |

The "hard labels" column is the one to watch. Several published process-reward
datasets label a prefix as good if *any* sampled completion from it reached
the right answer, which is an estimate of `1 - (1 - V*)^N` rather than of
`V*`, and is biased upward by an amount that grows with `N`. Measured bias
here: `[[N1u]]`.

*Measured by* [`experiments/prm_calibration.py`](experiments/prm_calibration.py).

---

## 2. Does it find the step where things went wrong?

A benchmark of traces with exactly one injected error at a known index. The
necessary and usually-skipped filter: an injected error only counts if it
actually damages the trace, which is checked by measuring `V*` on both sides
of it. Models self-correct, and an "error" the model routes around gives a
label that is simply wrong.

* injections discarded because `V*` did not drop: `[[N2a]]`
* exact first-error identification, PRM: `[[N2b]]`
* within one step: `[[N2c]]`
* best degenerate baseline (last step / longest step / uniform random),
  and which one: `[[N2d]]`

A step-localisation number that does not beat "guess the last step" by a clear
margin is not evidence of credit assignment, which is why that baseline is in
the table rather than in a footnote.

*Measured by* [`experiments/step_localization.py`](experiments/step_localization.py).

---

## 3. The frontier, at matched compute

Every search method on one engine, one accounting, one axis. The x-axis counts
generated tokens *and* estimated FLOPs including every reward-model and
verifier forward pass, because those are the costs that change the ordering.

| method | accuracy at 1x budget | at 8x | at 64x | tokens to reach the 8x accuracy of best-of-N |
|---|---|---|---|---|
| single sample | `[[N3a]]` | — | — | — |
| self-consistency (majority) | `[[N3b]]` | `[[N3c]]` | `[[N3d]]` | `[[N3e]]` |
| best-of-N, ORM | `[[N3f]]` | `[[N3g]]` | `[[N3h]]` | `[[N3i]]` |
| best-of-N, PRM-min | `[[N3j]]` | `[[N3k]]` | `[[N3l]]` | `[[N3m]]` |
| step-level beam search | `[[N3n]]` | `[[N3o]]` | `[[N3p]]` | `[[N3q]]` |
| lookahead beam | `[[N3r]]` | `[[N3s]]` | `[[N3t]]` | `[[N3u]]` |
| MCTS (PUCT) | `[[N3v]]` | `[[N3w]]` | `[[N3x]]` | `[[N3y]]` |

And the two comparisons that the frontier makes possible and that a
samples-based axis makes impossible:

* **sequential against parallel** — more samples, or longer ones, at the same
  FLOPs: `[[N3z]]`
* **difficulty-aware allocation** against a uniform budget, with the
  difficulty estimate taken from a held-out pilot rather than from the test
  labels: `[[N3aa]]`

*Measured by* [`experiments/matched_compute.py`](experiments/matched_compute.py).

---

## 4. Does the reinforcement learning add anything?

Group-relative policy optimisation against verifiable rewards, implemented
here rather than imported, on a small model with adapters. The claim under
test is not "pass@1 improves" — it does — but whether the improvement is new
capability or a redistribution of probability mass the model already had.

| | before | after |
|---|---|---|
| pass@1 | `[[N4a]]` | `[[N4b]]` |
| pass@8 | `[[N4c]]` | `[[N4d]]` |
| pass@64 | `[[N4e]]` | `[[N4f]]` |
| distinct correct solutions per solved problem | `[[N4g]]` | `[[N4h]]` |
| generation entropy | `[[N4i]]` | `[[N4j]]` |
| mean response length | `[[N4k]]` | `[[N4l]]` |

If pass@1 rises while pass@64 does not, the training sharpened the
distribution rather than extending it — which is a different and much less
interesting claim than the one usually made, and it is directly checkable at
matched sampling budget. Measured outcome: `[[N4m]]`.

Two implementation choices are reported as ablations rather than settled by
citation, because both are known to introduce artefacts: dividing the
group advantage by its standard deviation (which blows up on prompts where
every sample agrees), and normalising the loss by response length (which
creates a length bias). Their measured effect: `[[N4n]]`.

*Measured by* [`experiments/grpo_training.py`](experiments/grpo_training.py).

---

## 5. The parts everything else rests on

**The engine.** Reasoning traces form a tree, and a tree of traces shares
prefixes. The engine keeps one blocked KV cache for the whole tree with
copy-on-write branching, so expanding a node costs the new step only. Measured
against re-prefilling each branch independently:

* speedup at branching factor 4, depth 8: `[[N5a]]`
* peak memory against the independent-sequence baseline: `[[N5b]]`
* maximum logit deviation from the re-prefill path: `[[N5c]]`

The last number is the one that matters. A cache optimisation that changes the
logits is not an optimisation, and every search result in this repository is
downstream of it being small.

**The verifier.** Every label in the project inherits the answer checker's
error rate, so it is measured rather than assumed: false-accept `[[N5d]]`,
false-reject `[[N5e]]`, against hand-labelled samples. A 2% false-accept rate
puts a floor under every calibration number in section 1, and that floor is
quoted alongside them.

**The ground truth.** `V*` for `[[N5f]]` prefixes, median Wilson half-width
`[[N5g]]`, at a total cost of `[[N5h]]` generated tokens — `[[N5i]]` less than
a fixed-`N` allocation, because variance is `p(1-p)` and prefixes near 0 or 1
need far fewer rollouts to pin down.

**Contamination.** 13-gram overlap between the evaluation sets and everything
used for training, plus a completion probe: `[[N5j]]`.

*Measured by* [`experiments/tree_engine.py`](experiments/tree_engine.py),
[`verifier_audit.py`](experiments/verifier_audit.py),
[`value_ground_truth.py`](experiments/value_ground_truth.py) and
[`contamination.py`](experiments/contamination.py).

---

## Layout

```
src/arborist/
  engine/      the policy wrapper, step segmentation, and the trace tree with
               a blocked copy-on-write KV cache; exact compute accounting
  verify/      answer extraction and equivalence for maths, a resource-limited
               sandbox for code, and the audit of both
  value/       Monte-Carlo V* with Wilson intervals, adaptive rollout
               allocation, and the dataset builder
  reward/      the process and outcome reward models, their losses, and the
               calibration machinery
  search/      self-consistency, best-of-N, step beam, lookahead beam, MCTS,
               and the difficulty-aware budget allocator
  inject/      controlled single-step error injection and its V*-based filter
  rl/          adapters, rollout collection and group-relative policy
               optimisation, written out rather than imported
  eval/        pass@k, the matched-compute harness, risk-coverage, and the
               contamination audit
experiments/   one script per measurement above; writes results/*.json + *.png
benchmarks/    the shared-cache cross-check against independent re-prefill
examples/      short, runnable, public-API-only walkthroughs
docs/          SPEC.md, BUILD_PLAN.md, EXPERIMENTS.md, ENGINEERING.md
prompts/       the staged build transcript
```

## Running it

```bash
pip install -e ".[dev,verify,plot,data]"
pytest tests/ -q
```

```bash
PYTHONPATH=experiments python experiments/matched_compute.py
```

Every experiment takes `--quick` for a short run that exercises the whole
pipeline without producing a publishable number; CI runs all of them that way
on every push against tiny locally-constructed models, with the network
disabled, so no figure in this README can go stale without the build going red.

## A note on method

No number appears in this README until the experiment that produces it has
been run; until then it is a bracketed placeholder. One of those is
embarrassing in
a way that a plausible invented number is not, which is the only reason the
convention holds.

The repository's one real commitment is that a claim about reasoning should be
checkable against something that could contradict it — a Monte-Carlo estimate
with an interval for a reward model's score, hand labels for a verifier,
independent re-prefill for a cache, a FLOP count for a compute comparison,
pass@k for a claim about capability. Where a measurement disagreed with the
expected answer, the disagreement is in the write-up, not smoothed out of it.

## License

MIT
