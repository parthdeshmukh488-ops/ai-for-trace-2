Read AGENT_BRIEF.md. Phase 3's gate must be green and the corpus must exist.
This is phase 4 of docs/BUILD_PLAN.md: reward models.

Implement SPEC §4: `reward/heads.py`, `losses.py`, `prm.py`, `orm.py`,
`calibrate.py`.

The three objectives are all implemented, not just the good one:

* **binomial** — the correct objective, weighting each label by the confidence
  its rollout count supports;
* **MSE on the point estimate** — treats a label from 4 rollouts the same as
  one from 64;
* **hard labels** — positive if any rollout succeeded. This is what several
  published process-reward corpora use and it is biased: it estimates
  `1-(1-V*)^n`, not `V*`. For `V* = 0.1` and `n = 16` that is 0.81.

Check the bias arithmetic in closed form in a test before measuring it on
data, so that when the experiment reports a bias you know the predicted value
it should be near.

For calibration, the two defaults that are not negotiable:

* **equal-mass bins**, not equal-width. Equal-width bins put nearly all the
  mass in one or two bins when the predictor is confident, which is the usual
  case, and the resulting ECE is dominated by near-empty bins.
* **bootstrap over problems**, not over prefixes. Prefixes within a problem
  are correlated and bootstrapping over them understates the interval.

Implement the full Murphy decomposition and report resolution alongside
reliability. A model that predicts the base rate for everything has perfect
reliability, zero resolution and no value, and reporting calibration error
alone cannot tell the two apart. `test_constant_predictor_has_zero_resolution`
is the guard.

Train both configurations — frozen base with a trained head, and base with
low-rank adapters — three seeds each, and report the spread. At this scale a
single-seed comparison between objectives is noise.

Then run `experiments/prm_calibration.py` per docs/EXPERIMENTS.md#f4. Quote
phase 2's measured false-accept rate beside every ECE: a reward model trained
on labels that are wrong `e` of the time cannot be calibrated better than `e`.

Fill in `[[N1a]]` through `[[N1u]]` in README.md.

Gate:

```bash
pytest tests/test_reward.py tests/test_calibrate.py -q
PYTHONPATH=experiments python experiments/prm_calibration.py
grep -c '\[\[N1' README.md    # must print 0
```

Report: the filled table; the measured hard-label bias against its analytic
prediction; whether the binomial objective beat MSE by more than the seed
spread; whether adapters beat the frozen head by enough to justify the cost;
and how self-consistency compared, which the hypothesis says will be
competitive on calibration and worse on resolution.
