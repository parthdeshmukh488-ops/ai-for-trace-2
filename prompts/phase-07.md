Read AGENT_BRIEF.md. Phases 4 and 6 must be green. This is phase 7 of
docs/BUILD_PLAN.md, and it produces the project's central result.

Implement `eval/passk.py` and `eval/matched.py` per SPEC §7.1 and §7.2, then
run `experiments/matched_compute.py` per docs/EXPERIMENTS.md#f6.

pass@k uses the unbiased estimator, computed as
`1 - prod_{i=n-c+1}^{n} (1 - k/i)` for stability. Implementing it as "did any
of the first k samples succeed" is biased and much noisier; check against
exhaustive enumeration on small cases, and demonstrate the naive estimator's
bias in a test so the choice is justified by a number rather than by citation.

For the experiment, the parts that are not negotiable:

* **Both splits reported separately.** Pooling the easier and harder sets
  would let the easy one carry the conclusion.
* **Both axes plotted.** Generated tokens and FLOPs. The orderings differ
  whenever one method leans on the reward model more than another, and that
  divergence is one of the pre-registered hypotheses.
* **Paired bootstrap over problems**, 1000 resamples, with every comparison
  reported as a difference and an interval. Methods whose intervals overlap
  are reported as not distinguishable, not ranked. The report has a guard for
  this; do not route around it.
* **The sequential-against-parallel arm**: at each budget, `n` samples of
  length `L` against `n/2` samples at twice the token limit. This comparison
  is only possible on a compute axis, which is the point.
* **Difficulty-aware allocation** with fit and apply splits disjoint, and the
  leaked version computed too, so the size of the leakage is on record rather
  than merely warned about.

This is the most expensive experiment in the project. If the budget will not
cover it, reduce to 150 problems per split and record that in `scope`; do not
drop budgets or methods, because the shape of the frontier is the result.

Fill in `[[N3a]]` through `[[N3aa]]` in README.md.

Gate:

```bash
PYTHONPATH=experiments python experiments/matched_compute.py
grep -c '\[\[N3' README.md    # must print 0
```

Report: the frontier table; which method wins at each budget and whether the
win survives its interval; whether the ordering changed between the token axis
and the FLOP axis, and for which pair; the sequential-against-parallel result;
the allocation gain with and without leakage; and which of hypotheses (a)
through (f) survived.
