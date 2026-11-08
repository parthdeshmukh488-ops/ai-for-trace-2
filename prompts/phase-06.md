Read AGENT_BRIEF.md. Phase 4's gate must be green. This is phase 6 of
docs/BUILD_PLAN.md: search.

Implement SPEC §5: `search/consistency.py`, `bon.py`, `beam.py`, `mcts.py`.

The interface decision that makes phase 7 possible: every method takes a
**token budget**, not a sample count, and reports the budget it actually
consumed through `Budget`. A method that overspends by more than 5% is a
failed run rather than a data point, and the check belongs at the point of
deciding whether to expand, not after generating.

Implementation notes:

* All four best-of-N aggregations — `min`, `last`, `mean`, `prod`. They answer
  different questions and which wins is empirical; phase 7 compares them.
  `test_bon_aggregations_differ_on_a_constructed_case` guards against two of
  them being accidentally the same code path.
* MCTS uses PUCT with the prior taken as a softmax over the **sampled**
  children's mean token log-probability, renormalised over those children. The
  docstring must say this is a sampled prior and not the policy's true
  distribution over steps, which is not available. Mean backup by default,
  `max` as an ablation.
* `test_mcts_visits_concentrate_on_high_value_branch` with a synthetic scorer
  whose best branch is known is the test that catches selection or backup
  wired backwards. Write it before tuning anything.
* Reward-model forwards count toward the budget. This is where the usual
  comparisons go wrong, and `test_every_method_reports_through_budget` is the
  guard.

Also implement `search/allocator.py`. The difficulty model is fitted on a
calibration split and applied to the test split, and `allocate_budget` asserts
the two id sets are disjoint. Using test outcomes to estimate test difficulty
produces a large and entirely spurious gain and is the easiest way to get
phase 7's allocation arm wrong.

Then `examples/03_search_at_matched_compute.py`.

Gate:

```bash
pytest tests/test_search.py -q
```

Report: the gate output; the realised-against-requested budget for each method
on a sample workload; and, for MCTS, the tree sizes and maximum depths reached
at a few simulation counts, so phase 7's numbers can be sanity-checked against
them.
