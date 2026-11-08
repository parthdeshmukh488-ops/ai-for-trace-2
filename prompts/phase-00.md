Read AGENT_BRIEF.md and docs/SPEC.md before starting. This is phase 0 of
docs/BUILD_PLAN.md.

Set up the skeleton so that later phases have somewhere to land and CI is
green from the first commit.

Build:

1. `src/arborist/__init__.py` declaring the public surface from SPEC §0. Names
   that do not exist yet should be stubs raising
   `NotImplementedError("phase N")` naming the phase that implements them.
2. `src/arborist/engine/budget.py` — `Budget` and the analytic FLOP model from
   SPEC §1.4. Derive the per-token arithmetic in the docstring rather than
   quoting a formula, and include the unembedding term and the quadratic
   attention term explicitly. This is the x-axis of the project's headline
   result; an error here rescales every comparison in the repository.
3. `src/arborist/eval/stats.py` — a bootstrap over an explicit grouping key, a
   paired bootstrap for methods evaluated on the same problems, and a
   permutation test. Grouping is a required argument, not an option: almost
   every quantity in this project is correlated within a problem.
4. Seeding helpers. Per docs/ENGINEERING.md#determinism, the important
   decision is that each sequence gets its own generator seeded from
   `(seed, node_id)`, so sampling does not depend on batch composition. Make
   that decision now, in code, because retrofitting it later means re-running
   every experiment.
5. `tests/conftest.py` with the tiny randomly-initialised causal LM and
   trivial tokenizer from SPEC §8 — 2 layers, 2 heads, width 32, vocab 64 —
   plus a seeded generator fixture and `tmp_results`. Every later phase's
   tests depend on this fixture, so make it genuinely usable: it must generate
   text, expose a KV cache, and accept the engine's sampling parameters.

Write the phase-0 tests from docs/BUILD_PLAN.md. `test_flops_matches_hand_
computation` and `test_bootstrap_interval_coverage` are the two that matter:
both validate a measuring instrument before it is used, which is the pattern
the whole repository is built on.

Do not implement the policy wrapper or the trace tree. If a test seems to need
them, it belongs in phase 1.

One more test, which exists for the coverage floor as much as for its own
sake: `test_unimplemented_stubs_name_their_phase` asserts that every stub
in the public surface raises `NotImplementedError` with a message naming
the phase that will implement it. It covers those lines, so the floor holds
from the first commit, and it guards the promise that an early call to
something unbuilt says what is missing rather than failing obscurely.

Gate:

```bash
pytest tests/ -q && ruff check .
```

Report: the gate output, your FLOP derivation and the hand-computed value you
checked it against, the batch-independent sampling design you chose, and
anything in SPEC §0, §1.4 or §8 ambiguous enough that you had to decide.
