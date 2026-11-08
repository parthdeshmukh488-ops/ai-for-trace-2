Read AGENT_BRIEF.md. Phase 2's gate must be green. This is phase 3 of
docs/BUILD_PLAN.md: the ground truth. It is the hinge of the project.

Two things make this phase different from the others. The corpus it builds is
read by phases 4, 5 and 8, so a mistake in its sampling conditions means
regenerating it rather than patching it. And the estimator it relies on has to
be validated before the generation budget is spent, not after.

So do it in this order.

**First, validate the instrument.** Implement `value/wilson.py` including the
simulation harness from SPEC §3.4, and run it before writing anything else.
True `p` on a 21-point grid, `n` in {4, 8, 16, 32, 64}, 20000 replicates, for
the fixed-`n` interval and for the adaptive stopping rule. Measure the
**realised** coverage of the nominal 95% interval in both cases.

The adaptive rule is expected to under-cover: stopping when an interval
becomes narrow is optional stopping and it breaks the interval's guarantee.
That shortfall is a real finding and it produces a widening factor that the
corpus must apply before anything downstream uses it. Do not skip this because
the interval formula is textbook — the formula is fine; the stopping rule
wrapped around it is the problem.

**Then** implement `value/montecarlo.py`, `value/allocate.py` and
`value/dataset.py` per SPEC §3.2, §3.3 and §3.5. Points to be careful about:

* `ValueEstimate` carries its `SamplingConditions` and the serialiser refuses
  to write one without them. Make producing a bare `V*` awkward by design.
* The step-advantage interval is **not** the difference of two interval
  endpoints. Use Newcombe's interval for the difference of independent
  proportions, or a bootstrap over the pooled rollouts; say which, and
  validate its coverage by simulation too.
* Splits are by **problem**, never by prefix. Prefixes within a problem are
  heavily correlated and a prefix-level split leaks difficulty into the test
  set, inflating every reward-model number in phase 4. Assert disjointness.
* The empty prefix's `V*` is the model's unaided pass rate and is the baseline
  every step advantage is measured against. Include it.

Then run `experiments/value_ground_truth.py` per docs/EXPERIMENTS.md#f2.
Confirm the sampling temperature and the model revision before starting the
generation; both are baked into the corpus.

Fill in `[[N5f]]` through `[[N5i]]` in README.md: the number of prefixes in
the corpus, the median Wilson half-width, the total tokens spent, and the
saving against a fixed `N = 32` allocation.

Gate:

```bash
pytest tests/test_wilson.py tests/test_value.py -q
PYTHONPATH=experiments python experiments/value_ground_truth.py
```

Report: realised coverage for both allocation rules and the widening factor;
the corpus size; median and worst half-width achieved; the allocation
histogram; total tokens spent and the saving against fixed `N = 32`, computed
from realised completion lengths rather than an assumed mean; and the exact
sampling conditions the corpus is conditioned on.
