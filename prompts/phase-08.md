Read AGENT_BRIEF.md. Phase 3's corpus must exist and phase 7's pass@k
implementation is needed here. This is phase 8 of docs/BUILD_PLAN.md:
reinforcement learning.

Implement SPEC §6: `rl/lora.py`, `rollout.py`, `grpo.py`, `diagnostics.py`.

Write the adapter out rather than importing one. It is forty lines, and the
`alpha / r` scaling convention is exactly the detail that differs between
implementations and silently rescales the effective learning rate.

For the policy-gradient step, three things in order of how invisible they are
when wrong:

1. **The importance ratio must be exactly 1 on the first inner step.** If it
   is not, the behaviour log-probabilities were recomputed under the updated
   policy instead of being cached from the rollout, and the first update is
   wrong in a way nothing else will reveal.
   `test_ratio_is_one_at_first_inner_step` is the guard and it is not optional.
2. **The KL estimator is `ratio - log(ratio) - 1`**, which is non-negative per
   sample. `-log(ratio)` is not, and a KL term that goes negative makes the
   training dynamics incomprehensible.
3. **Zero-variance groups.** On an easy prompt every sample gets the same
   reward and the normalised advantage divides by zero. Handle it explicitly
   and report the fraction of groups affected — on an easy dataset it is a
   large fraction, which is the reason the centred variant exists.

Implement both advantage variants and both loss normalisers, and treat them as
ablations to measure rather than choices to settle by citation. Sequence-level
normalisation weights short sequences' tokens more heavily and introduces a
length bias whose direction depends on the advantage's sign; token-level is
the default.

Log every diagnostic in SPEC §6.3 at every step. Entropy collapse and length
inflation are the standard failure modes and they are visible in these and not
in the reward curve.

Then run `experiments/grpo_training.py` per docs/EXPERIMENTS.md#f7.

The question is not whether pass@1 improves — it will. It is whether pass@64
improves. Measure both at the **same** sampling temperature before and after:
lowering temperature raises pass@1 and lowers pass@k by itself, so a
temperature difference between the two measurements would answer the question
spuriously. Share the "before" evaluation across arms rather than recomputing
it.

Fill in `[[N4a]]` through `[[N4n]]` in README.md.

Gate:

```bash
pytest tests/test_grpo.py -q
PYTHONPATH=experiments python experiments/grpo_training.py
grep -c '\[\[N4' README.md    # must print 0
```

Report: pass@1, pass@8 and pass@64 before and after; the direct answer —
sharpening or capability — by the pass@k criterion; the entropy and
distinct-solution trajectories; the fraction of zero-variance groups; and the
measured effect of the advantage and normaliser choices, with the caveat that
two seeds per arm cannot establish a small difference.
