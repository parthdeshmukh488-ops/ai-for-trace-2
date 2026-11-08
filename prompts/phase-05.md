Read AGENT_BRIEF.md. Phase 4's gate must be green. This is phase 5 of
docs/BUILD_PLAN.md: credit assignment.

Implement `inject/perturb.py` and `inject/bench.py`, then run
`experiments/step_localization.py` per docs/EXPERIMENTS.md#f5.

Six error types, injected at a uniformly sampled step index of an otherwise
correct trace: arithmetic slip, dropped constraint, wrong formula, sign error,
off-by-one, and a valid-but-non-advancing step. The last one is the
interesting case — it tests whether the scorer penalises lack of progress or
only outright error.

**The filter is the whole experiment.** An injected error only counts if it
actually damaged the trace, which is checked by measuring `V*` on both sides
of the injection and requiring a drop larger than the combined interval width.
Models self-correct, and an "error" the model routes around gives a
first-error label that is simply wrong. Report the discard rate **per error
type**: some types will be recovered from far more often than others, and that
is itself a result about what the model can repair.

Report the degenerate baselines in the main table, not a footnote: always the
last step, always the longest step, uniform random. Step indices of errors in
generated traces are not uniformly distributed, so "guess the last step" is
stronger than it sounds, and a localisation number that does not clearly beat
it is not evidence of credit assignment.

Also compute the oracle upper bound: the localisation achievable by a scorer
that sees `V*` itself. That bounds what any scorer of this kind could do and
puts the PRM's number in proportion.

Fill in `[[N2a]]` through `[[N2d]]` in README.md.

Gate:

```bash
pytest tests/test_inject.py -q
PYTHONPATH=experiments python experiments/step_localization.py
grep -c '\[\[N2' README.md    # must print 0
```

Report: the discard rate overall and per error type; exact and within-one-step
localisation; the best degenerate baseline and which it was; the margin
between the PRM and that baseline, stated plainly even if it is small; and the
oracle bound.
