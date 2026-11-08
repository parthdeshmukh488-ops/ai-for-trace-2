# Build plan

Eleven phases. Each has deliverables, the tests that must exist, and an **exit
gate** that is a command with a pass/fail outcome rather than a judgement call.
Do not begin a phase until the previous gate is green.

Effort estimates assume someone who knows the material. They are there to flag
when a phase has gone wrong, not as a target. Wall-clock for the rollout-heavy
phases is dominated by generation, not by implementation, and the estimates
separate the two.

---

## Phase 0 — bootstrap · ~half a day

**Deliverables**

* `src/arborist/__init__.py` with the public surface from SPEC §0 declared;
  names not yet implemented are stubs raising
  `NotImplementedError("phase N")`.
* `engine/budget.py` — `Budget` and the analytic FLOP model from SPEC §1.4,
  with the arithmetic derived in the docstring rather than quoted.
* `eval/stats.py` — bootstrap intervals over a grouping key, the paired
  bootstrap, and a permutation test.
* `util` seeding helpers: explicit generators, no global RNG in library code.
* `tests/conftest.py` with the tiny randomly-initialised causal LM and trivial
  tokenizer from SPEC §8, a seeded generator fixture, and `tmp_results`.

**Tests**

* `test_flops_matches_hand_computation` — one small config worked out by hand
  in the test body. The FLOP model is the x-axis of the project's headline
  result; an error here rescales every comparison.
* `test_bootstrap_interval_coverage` — simulate from a known distribution and
  check the nominal 95% interval covers about 95% of the time. The measuring
  device gets measured first, here as everywhere.
* `test_paired_bootstrap_detects_known_difference`.
* `test_tiny_model_generates` — the fixture is load-bearing for every later
  phase's tests.

**Gate**

```bash
pytest tests/ -q && ruff check .
```

---

## Phase 1 — the engine · ~4 days

**Deliverables**

* `engine/policy.py`, `engine/segment.py`, `engine/kvtree.py` per SPEC §1.
* `benchmarks/cross_check_cache.py` per SPEC §1.3.
* `experiments/tree_engine.py` (§F1).
* `examples/01_trace_tree.py`.

**Tests that must exist**

| test | what would fail it |
|---|---|
| `test_shared_cache_matches_reprefill` | the core exactness claim; `allclose` at the stated tolerance on tiny models |
| `test_position_ids_follow_path_length` | positions computed from block index rather than path depth — the most likely bug in this component |
| `test_branch_does_not_mutate_parent_blocks` | copy-on-write violated; a sibling's tokens appearing in another's context |
| `test_partial_last_block_is_read_only_after_branch` | two children writing into a shared partial block |
| `test_evict_then_recompute_is_exact` | eviction losing state instead of costing time |
| `test_evict_never_frees_referenced_blocks` | an in-use block being reclaimed |
| `test_budget_counts_recomputation` | recomputation vanishing from the accounting |
| `test_segmentation_reports_fallback_rate` | a silently-unsegmented corpus |
| `test_block_waste_is_reported` | the branch-point waste going unmeasured |

**Gate**

```bash
pytest tests/ -q -m "not slow and not network"
python benchmarks/cross_check_cache.py
PYTHONPATH=experiments python experiments/tree_engine.py --quick
```

The cross-check must print a maximum absolute logit deviation under `1e-3` and
write `benchmarks/results/cross_check_cache.json`.

**Report** the achieved deviation, the measured speedup against re-prefill at
several branching factors and depths, the peak-memory comparison, the measured
block waste, and the ratio of analytic FLOPs to measured throughput. If that
ratio is far from 1, say so — every later FLOP axis inherits it.

---

## Phase 2 — verification · ~2 days

**Deliverables**

* `verify/answer.py`, `verify/sandbox.py`, `verify/audit.py` per SPEC §2.
* `experiments/verifier_audit.py` (§F3).

**Tests**

| test | what would fail it |
|---|---|
| `test_extraction_rule_precedence` | a `\boxed` answer losing to the last-number rule |
| `test_normalisation_equivalences` | a table of pairs that must and must not match |
| `test_symbolic_timeout_is_enforced` | a hanging equivalence check |
| `test_sandbox_cpu_limit_stops_infinite_loop` | the limit not firing, or the wall clock firing first |
| `test_sandbox_memory_limit_stops_allocation` | as above for `RLIMIT_AS` |
| `test_sandbox_fork_limit_stops_process_bomb` | `RLIMIT_NPROC` not set |
| `test_sandbox_scratch_dir_is_removed` | leaked temp directories |
| `test_audit_intervals_are_wilson` | a bare rate quoted on 200 samples |

The sandbox tests are marked `sandbox` and each must assert *which* limit
stopped the program, not merely that it was stopped.

**Gate**

```bash
pytest tests/test_answer.py tests/test_sandbox.py -q
PYTHONPATH=experiments python experiments/verifier_audit.py
```

**Report** the false-accept and false-reject rates with intervals, the
stratified sampling procedure used, the fraction of extractions that fell
through to the last-number rule, and the symbolic-timeout hit rate. The
false-accept rate is the floor under every calibration number in phase 4, so
record it where phase 4 will find it.

---

## Phase 3 — the ground truth · ~2 days of work, plus generation time

**Deliverables**

* `value/wilson.py` including the coverage simulation harness,
  `value/montecarlo.py`, `value/allocate.py`, `value/dataset.py` per SPEC §3.
* `experiments/value_ground_truth.py` (§F2).
* `examples/02_ground_truth_value.py`.

**Tests**

| test | what would fail it |
|---|---|
| `test_wilson_matches_closed_form` | an algebra slip in the interval |
| `test_wilson_coverage_by_simulation` | an interval that does not cover at its nominal rate |
| `test_adaptive_coverage_by_simulation` | optional stopping degrading coverage without the correction |
| `test_difference_interval_is_not_endpoint_subtraction` | the 40%-too-wide interval for step advantage |
| `test_conditions_are_required` | a `ValueEstimate` serialising without its sampling conditions |
| `test_corpus_split_is_by_problem` | prefix-level splitting, which leaks difficulty |
| `test_empty_prefix_value_equals_unaided_pass_rate` | the baseline every advantage is measured against |

`test_adaptive_coverage_by_simulation` is the important one. It is expected to
*fail* at the nominal rate — that is the point. It asserts the measured
coverage against the empirically determined value and records the widening
factor the corpus must apply.

**Gate**

```bash
pytest tests/test_wilson.py tests/test_value.py -q
PYTHONPATH=experiments python experiments/value_ground_truth.py
```

**Report** the corpus size, median and worst interval half-width, total tokens
spent, the saving against fixed-`N`, and the measured coverage of both
allocation procedures with the widening factor applied downstream.

---

## Phase 4 — reward models · ~3 days plus training time

**Deliverables**

* `reward/heads.py`, `losses.py`, `prm.py`, `orm.py`, `calibrate.py` per
  SPEC §4.
* `experiments/prm_calibration.py` (§F4).

**Tests**

| test | what would fail it |
|---|---|
| `test_binomial_loss_gradient_sign` | the objective pushing the wrong way |
| `test_binomial_loss_weights_by_n` | a prefix with `n=64` pulling no harder than one with `n=4` |
| `test_hard_label_bias_is_reproduced_analytically` | the `1-(1-p)^n` arithmetic, checked in closed form before it is measured on data |
| `test_ece_equal_mass_bins_have_equal_counts` | equal-width bins defaulted in |
| `test_brier_decomposition_sums_to_brier` | an algebra error in the Murphy identity |
| `test_constant_predictor_has_zero_resolution` | the metric failing to expose the degenerate case it exists to expose |
| `test_bootstrap_groups_by_problem` | intervals computed over correlated prefixes |

**Gate**

```bash
pytest tests/test_reward.py tests/test_calibrate.py -q
PYTHONPATH=experiments python experiments/prm_calibration.py
```

Fill in `[[N1a]]`–`[[N1u]]` in README.md.

**Report** the full calibration table, the measured hard-label bias, whether
the binomial objective beat MSE and by how much relative to its interval,
whether the adapted base model beat the frozen one enough to justify its cost,
and the verifier's false-accept floor quoted alongside every ECE.

---

## Phase 5 — credit assignment · ~2 days plus rollouts

**Deliverables**

* `inject/perturb.py`, `inject/bench.py` per SPEC §2 of the README and the
  protocol in EXPERIMENTS §F5.
* `experiments/step_localization.py` (§F5).

**Tests**

| test | what would fail it |
|---|---|
| `test_injection_changes_exactly_one_step` | a perturbation leaking into neighbouring steps |
| `test_injection_filter_requires_value_drop` | unfiltered injections, whose first-error labels are simply wrong |
| `test_degenerate_baselines_are_computed` | the experiment shipping without the baseline that decides whether it means anything |
| `test_localisation_metrics_on_synthetic_scores` | metric bugs, checked against hand-built score sequences with known answers |

**Gate**

```bash
pytest tests/test_inject.py -q
PYTHONPATH=experiments python experiments/step_localization.py
```

Fill in `[[N2a]]`–`[[N2d]]`.

**Report** the discard rate of the `V*` filter, exact and within-one-step
localisation rates, the best degenerate baseline and its score, and the margin
between them. If the margin is small, that is the finding and the README leads
with it.

---

## Phase 6 — search · ~4 days

**Deliverables**

* `search/consistency.py`, `bon.py`, `beam.py`, `mcts.py` per SPEC §5.
* `examples/03_search_at_matched_compute.py`.

**Tests**

| test | what would fail it |
|---|---|
| `test_every_method_respects_its_token_budget` | a method overspending, which would invalidate every matched-compute point |
| `test_every_method_reports_through_budget` | an uncounted reward-model call |
| `test_bon_aggregations_differ_on_a_constructed_case` | an aggregation that is secretly the same as another |
| `test_mcts_prior_is_renormalised_over_sampled_children` | a prior that does not sum to one |
| `test_mcts_visits_concentrate_on_high_value_branch` | selection or backup wired wrongly; use a synthetic scorer with a known best branch |
| `test_beam_keeps_exactly_width_candidates` | silent beam growth |
| `test_allocator_asserts_disjoint_fit_and_apply_ids` | the leakage that makes difficulty-aware allocation look free |

**Gate**

```bash
pytest tests/test_search.py -q
```

---

## Phase 7 — the frontier · ~2 days plus substantial generation

**Deliverables**

* `search/allocator.py`, `eval/passk.py`, `eval/matched.py` per SPEC §5, §7.1,
  §7.2.
* `experiments/matched_compute.py` (§F6) — the project's central result.

**Tests**

* `test_passk_matches_enumeration` — the closed form against exhaustive
  enumeration on small `n`, `c`, `k`.
* `test_passk_naive_estimator_is_biased` — demonstrate the bias the unbiased
  estimator exists to avoid, so the choice is justified by a number.
* `test_curve_reports_both_axes`.
* `test_overlapping_intervals_are_not_ranked` — the report's own guard against
  ordering methods it cannot distinguish.

**Gate**

```bash
PYTHONPATH=experiments python experiments/matched_compute.py
```

Fill in `[[N3a]]`–`[[N3aa]]`.

**Report** the frontier table; which method wins at each budget and whether
the win survives its interval; whether the ordering changes between the token
axis and the FLOP axis; the sequential-against-parallel result; and the
difficulty-aware allocation gain with the confirmation that fit and apply
splits were disjoint.

---

## Phase 8 — reinforcement learning · ~4 days plus training time

**Deliverables**

* `rl/lora.py`, `rollout.py`, `grpo.py`, `diagnostics.py` per SPEC §6.
* `experiments/grpo_training.py` (§F7).

**Tests**

| test | what would fail it |
|---|---|
| `test_lora_scaling_is_alpha_over_r` | the convention that silently rescales the learning rate |
| `test_base_parameters_are_frozen` | gradients reaching the base weights |
| `test_zero_variance_group_is_handled` | the normalised advantage exploding on a prompt where every sample agrees |
| `test_advantage_variants_differ_as_specified` | centred and normalised being the same code path |
| `test_token_and_sequence_normalisers_differ` | the length bias being invisible |
| `test_kl_estimator_is_non_negative` | `-log(ratio)` used in place of the k3 estimator |
| `test_ratio_is_one_at_first_inner_step` | an off-by-one in the behaviour-policy log-probs, which makes the first update silently wrong |
| `test_grpo_step_improves_a_toy_objective` | an end-to-end check on a task with a known optimum |

`test_ratio_is_one_at_first_inner_step` catches the most common and most
invisible bug in a policy-gradient implementation: on the first inner epoch the
importance ratio must be exactly one, and if it is not, the behaviour
log-probabilities were recomputed under the updated policy.

**Gate**

```bash
pytest tests/test_grpo.py -q
PYTHONPATH=experiments python experiments/grpo_training.py
```

Fill in `[[N4a]]`–`[[N4n]]`.

**Report** pass@1, pass@8 and pass@64 before and after at matched temperature;
the distinct-solution and entropy trajectories; whether the gain is sharpening
or capability by the pass@k criterion; and the measured effect of the advantage
and normaliser choices.

---

## Phase 9 — abstention and contamination · ~2 days

**Deliverables**

* `eval/selective.py`, `eval/contamination.py` per SPEC §7.3, §7.4.
* `experiments/selective_prediction.py` (§F8),
  `experiments/contamination.py` (§F9).

**Tests**

* `test_risk_coverage_endpoints` — coverage 1 gives the base error rate;
  coverage 0 is undefined and must raise rather than return zero.
* `test_aurc_on_a_perfect_and_a_useless_signal`.
* `test_ngram_overlap_finds_a_planted_duplicate`.
* `test_completion_probe_on_a_memorised_string` — plant a string in the tiny
  model's training to confirm the probe can detect what it claims to.

**Gate**

```bash
PYTHONPATH=experiments python experiments/selective_prediction.py
PYTHONPATH=experiments python experiments/contamination.py
```

Fill in `[[N5j]]` and the remaining `[[N5x]]` placeholders.

---

## Phase 10 — the write-up · ~3 days

**Deliverables**

* Every `[[Nx]]` placeholder replaced by a measured value.
* `docs/EXPERIMENTS.md` complete: every `Result` block filled, every
  pre-registered hypothesis marked as surviving or not.
* A `## Where the measurements disagreed` section, in docs and summarised in
  the README.
* The README's prose rewritten around what was found rather than what was
  planned, keeping the structure and the voice.
* `docs/LIMITATIONS.md`: what `V*` under one policy does not tell you about
  another; what a 1.5B model on grade-school maths does not tell you about
  frontier models on hard problems; how much of the frontier result depends on
  the FLOP model being right; what the verifier's false-accept rate does to
  every number downstream. Written as the strongest honest case against the
  project's own conclusions.
* Coverage at or above 90%, enforced in CI. `ruff check .` clean.
* All three examples runnable from a clean checkout.

**Gate**

```bash
pytest tests/ -q -m "not gpu and not network" --cov=arborist --cov-fail-under=90
ruff check .
grep -c '\[\[N' README.md    # must print 0
```

---

## Dependency summary

```
  0 -> 1 -> 2 -> 3 -> 4 -> 5
                      |    |
                      +----+-> 6 -> 7
                      |
                      +-> 8
  7,8 -> 9 -> 10
```

Phase 3 is the hinge: the `V*` corpus is what phases 4, 5 and 8 all read, and
it is the one artefact expensive enough that building it twice hurts. Get the
coverage validation in phase 3 right before spending the generation budget.
