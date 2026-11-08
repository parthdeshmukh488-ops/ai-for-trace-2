Read AGENT_BRIEF.md. Phases 7 and 8 must be green. This is phase 9 of
docs/BUILD_PLAN.md: abstention and contamination.

Implement SPEC §7.3 and §7.4: `eval/selective.py` and `eval/contamination.py`.

For selective prediction, compare every confidence signal on one axis: the PRM
score under each aggregation, self-consistency agreement, mean token
log-probability, predictive entropy, and the ORM score. Report the
risk-coverage curve, AURC with a bootstrap interval, and accuracy at 50% and
80% coverage. Coverage 0 is undefined and must raise rather than return zero.

For contamination, both checks, because they catch different things:

* 13-gram overlap between the evaluation problems and everything this project
  trained on. Validate the checker with a planted duplicate first — an overlap
  check that cannot find a known match is worse than none.
* The completion probe: give the first half of a problem statement, sample
  greedily, measure normalised edit distance to the true second half. This
  catches contamination that reached the model before this project did, which
  no overlap check against this project's data could find.

The probe needs a null to be interpretable: run it on freshly-written problems
in the same style that the model cannot have memorised, and report the
distribution for both. A high reproduction rate means nothing without knowing
what the rate is on text that is certainly unseen.

Report the numbers whatever they show. If the probe suggests contamination,
re-report phase 7's and phase 8's headline numbers on the lowest-suspicion
subset alongside the full set rather than instead of it.

Fill in `[[N5j]]` and any remaining `[[N5x]]` placeholders.

Gate:

```bash
PYTHONPATH=experiments python experiments/selective_prediction.py
PYTHONPATH=experiments python experiments/contamination.py
```

Report: the best confidence signal and its AURC interval; accuracy at 50% and
80% coverage; the overlap fraction; and the completion probe against its null,
with a plain statement of whether contamination is a live concern for this
project's conclusions.
