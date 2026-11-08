Read AGENT_BRIEF.md. All previous gates must be green. This is phase 10 of
docs/BUILD_PLAN.md: the write-up.

The implementation is done. What remains is making the repository say true
things about itself, which is the part that decides whether any of it was
worth doing.

1. **Every placeholder replaced.** `grep -c '\[\[N' README.md` prints 0. Each
   number traceable to a committed `experiments/results/*.json`. If a
   measurement was never run, remove the claim rather than leaving a
   placeholder or inventing a value.

2. **`docs/EXPERIMENTS.md` completed.** Every `Result` block filled, and for
   each pre-registered hypothesis an explicit statement of whether it survived.
   F6 has six, F7 has five; go through them one at a time.

3. **The `## Where the measurements disagreed` section written.** Candidates
   you should already have: whether the coverage of the adaptive allocator was
   what it claimed; whether the binomial objective's advantage survived the
   seed spread; whether step localisation beat "guess the last step"; whether
   the FLOP axis reordered any pair of search methods; whether pass@64 moved.
   If this section is empty, the experiments were not sharp enough, and saying
   so is better than an empty section.

4. **Rewrite the README's prose around what was found.** It currently argues
   for a programme; it should argue for a result. Keep the structure and the
   voice, change the claims to the measured ones. If a headline claim did not
   survive, the README leads with what did.

5. **`docs/LIMITATIONS.md`**, which does not exist yet and should. Write the
   strongest honest case against the project's own conclusions:

   * `V*` is defined relative to one policy at one temperature. What does a
     reward model calibrated against it tell you about any other policy?
   * A 1.5B model on grade-school mathematics is not a frontier model on hard
     problems. Which of the conclusions are about test-time compute and which
     are about this model's particular weaknesses?
   * How much of the frontier result depends on the analytic FLOP model being
     right, and what did F1 measure its error to be?
   * The verifier's false-accept rate puts a floor under every calibration
     number. Where is that floor close to the differences being reported?
   * Two seeds per arm in F7. Which of its conclusions are safe at that sample
     size and which are suggestive only?

   If you cannot make that case, you do not yet understand the results well
   enough to publish them.

6. **Examples run from a clean checkout.** All three, no manual setup beyond
   the install.

7. **Coverage and lint.** 90% floor enforced in CI, `ruff check .` clean.

Gate:

```bash
pytest tests/ -q -m "not gpu and not network" --cov=arborist --cov-fail-under=90
ruff check .
grep -c '\[\[N' README.md    # must print 0
```

Report: the gate output, `docs/LIMITATIONS.md` in summary, and the three
findings you would lead with if this were being presented. Then say which one
you are least confident in and what measurement would settle it.
