# Agent brief — read this before writing any code

This repository is specified but not built. You are implementing it. The
specification is complete enough that you should not need to invent scope; if
you think you do, the answer is in [`docs/SPEC.md`](docs/SPEC.md) or it is a
question worth asking rather than guessing.

Work through [`prompts/`](prompts/) in order. Each file is one phase with its
own exit gate. Do not start a phase before the previous phase's gate is green.

---

## 1. What this project is

A test-time-compute and reasoning library whose distinguishing property is
that its central quantity is **measured rather than assumed**. In a verifiable
domain the value of a partial reasoning trace,

```
  V*(prefix) = P( final answer correct | prefix, policy, temperature )
```

is a binomial proportion with a confidence interval. Everything here follows
from treating that as the ground truth: a process reward model is scored
against it, a search method is compared on a compute axis rather than a sample
count, and a reinforcement-learning run is judged on whether it extends the
distribution or merely sharpens it.

The dependency order of the whole build:

```
  trace engine        ->  cross-checked against independent re-prefill
         |
  verifiers           ->  cross-checked against hand labels
         |
  Monte-Carlo V*      ->  carries its own Wilson intervals
         |
  reward models       ->  cross-checked against held-out V*
         |
  search + RL         ->  compared at matched FLOPs, with pass@k as the control
```

Nothing downstream is trustworthy if something upstream is not verified. In
particular: every label in this repository inherits the verifier's error rate,
and every search result inherits the cache's exactness. Build in order.

---

## 2. Hard rules

These are not stylistic. Breaking any of them makes the repository worthless
for its stated purpose.

**R1 — Never write a number you did not measure.** No example number, no
"typical" value, no plausible-looking result in a docstring, README, comment
or JSON file. The README ships with `[[Nx]]` placeholders; you replace a
placeholder only with the value your experiment actually printed, in the same
commit that produces the `results/*.json` it came from.

**R2 — Every `V*` carries its conditions.** Model, temperature, top-p, token
limit, `N`, and the Wilson interval. `V*` is a property of a prefix *under a
sampling policy*, not of the prefix alone. A `V*` quoted bare is not a
measurement, and the API is built so that producing one is awkward: the value
type carries its conditions and the serialiser refuses to write it without
them.

**R3 — Compute comparisons count everything.** Reward-model forward passes,
verifier calls, the prefill of every branch, discarded samples. A method that
wins on samples and loses on FLOPs has lost. `engine/budget.py` is the single
source of truth and every search method reports through it; no experiment
estimates a cost it could have counted.

**R4 — Report disagreement, do not resolve it quietly.** If the cheap method
beats the sophisticated one, if a published result does not reproduce, if
pass@64 does not move — that is a finding and it goes in the write-up. The
existing portfolio's credibility rests on these being present.

**R5 — Determinism is a tested property.** Every stochastic entry point takes
an explicit `seed`. Sampling, rollout allocation, bootstrap resampling and
data shuffling all take generators explicitly; library code never reads global
RNG state. There is a test that runs each pipeline twice and asserts identical
output. See `docs/ENGINEERING.md#determinism`.

**R6 — The shared cache must be exact.** The trace tree's whole purpose is to
avoid recomputing shared prefixes, and the moment it changes the logits every
number in the repository is wrong. `benchmarks/cross_check_cache.py` compares
against independent re-prefill, CI runs it, and the measured deviation is
published rather than merely asserted to be small.

**R7 — The sandbox is a security boundary, not a convenience.** Generated code
runs in a subprocess with CPU-time, memory, file-descriptor and process limits
and no network. The limits are tested by programs that try to exceed each one.
Never execute generated code in the host process, not even for a quick check.

**R8 — Statistical claims carry intervals.** Accuracy differences come with
bootstrap confidence intervals and, where methods share problems, a paired
test. Two methods whose intervals overlap are not ranked, and the write-up
says they are not distinguishable at this sample size rather than reporting
the point estimates in order.

**R9 — Nothing about the tooling used to write this code appears in the
repository.** No tool names, no assistant attribution, no generated-by notices
in files, commit messages, or metadata. Commits are authored normally.

---

## 3. House style

Match the author's existing repositories. The style is specific and consistent.

**Module docstrings argue for the design.** They do not describe the file; they
explain why the file is the way it is, usually with a short bulleted case and
a named alternative that was rejected.

**Comments explain reasoning, especially for constants.** A magic number gets
a comment saying how it was chosen and what measurement justifies it. If you
chose it by trying things until it worked, say that.

**Prose wraps at 78 columns. Code is `line-length = 100`, ruff-clean.**

**`from __future__ import annotations` at the top of every module. `__all__`
is explicit and sorted.**

**Public API is small and flat.** Users should be able to do real work from
`arborist.<thing>` without reaching into submodules.

**Experiment scripts open with the question, not the method.** The docstring
states what is being measured, what it is compared against, why that
comparison is the informative one, and ends with the usage line:

```
    python experiments/matched_compute.py [--quick]
```

Every experiment:
- takes `--quick`, which must exercise every code path at reduced size using
  a tiny locally-constructed model and no network,
- takes `--seed`, defaulting to a fixed value,
- writes `experiments/results/<name>.json` with the raw numbers plus a
  `metadata` block (seed, git SHA, device, model id and revision, sampling
  parameters, token and FLOP totals, wall time, scope),
- writes `experiments/results/<name>.png` through `experiments/style.py`,
- prints the headline numbers to stdout in a form you can paste into a table,
- and under `--quick` writes `<name>_quick.json`, which is gitignored because
  it is a smoke test and never a result.

**Figures are plotted against the informative axis.** Accuracy against FLOPs,
not against `N`. Calibration as a reliability diagram with its interval band,
not as a single ECE number. Error localisation against the degenerate
baseline, not in isolation.

---

## 4. Definition of done, per component

A component is done when all five hold:

1. It is implemented and its public API matches `docs/SPEC.md`.
2. Its tests pass, including at least one test that would fail if the
   component were subtly wrong rather than obviously broken.
3. Its oracle cross-check passes.
4. Its docstrings argue for the design and contain no unmeasured numbers.
5. `pytest tests/ -q --cov=arborist --cov-fail-under=90` is green and
   `ruff check .` is clean.

---

## 5. Commits

Small, self-contained, one idea each. Present tense, lowercase after the first
word, no trailing period, no scope prefixes:

```
Add the blocked KV cache and its copy-on-write branching
Measure the hard-label bias in process reward targets
Fix the budget accounting; reward-model prefill was not counted
```

Commit the `results/*.json` and `results/*.png` that back a README number in
the same commit that fills in the placeholder. Never commit weights, adapters
or rollout corpora — `.gitignore` covers them, keep it that way.

---

## 6. Compute

This project generates tokens, and tokens cost time. `docs/ENGINEERING.md#compute-tiers`
gives the budget for every experiment at every tier, with the arithmetic shown
so you can recompute it for a different model.

The expensive phase is the `V*` dataset, and it is expensive once: everything
downstream reads it from disk. Build it with the adaptive allocator from the
start rather than with a fixed `N` you will regret.

If a run is too expensive to complete, reduce its *scope* explicitly and say
so in the results JSON (`"scope": "reduced: 200 of 500 problems, N=16 not 32"`)
rather than quietly running less and reporting it as the full measurement.

---

## 7. When you are stuck

The specification is wrong sometimes. If you find an inconsistency between
`docs/SPEC.md` and reality — an algorithm that cannot work as described, a
metric that is not well-defined, a ground truth that turns out not to be
ground truth — stop, write down the problem, fix the specification, and note
the change in `docs/SPEC.md` under a `## Revisions` heading at the bottom.
That is a better outcome than an implementation that quietly diverges from its
own documentation.
