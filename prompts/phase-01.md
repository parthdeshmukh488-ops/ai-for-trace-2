Read AGENT_BRIEF.md. Phase 0's gate must be green. This is phase 1 of
docs/BUILD_PLAN.md: the engine.

Implement SPEC §1: `engine/policy.py`, `engine/segment.py`, `engine/kvtree.py`.

Order of work, which matters:

1. `policy.py` — the batched sampling wrapper. No default temperature; every
   generation records its conditions. All token counting goes through `Budget`.
2. `segment.py` — step segmentation with the four policies, the fallback flag,
   and the accounting. Report steps per trace, tokens per step, and the
   fallback rate; a step-level project whose segmentation silently fails on a
   third of its corpus is measuring something other than steps.
3. `kvtree.py` — `BlockCache` and `TraceTree`. The things to get right rather
   than approximately right:
   - **Position ids come from path length, never from block index.** This is
     the most likely bug in the component and the hardest to see, because the
     model still produces fluent text with slightly wrong positions.
   - A child always starts a fresh block, so a parent's partially-filled last
     block becomes read-only at the branch point. Measure and report the
     wasted slots; the waste is the price of correctness here.
   - Eviction reclaims only unreferenced blocks, and an evicted node keeps its
     token ids so it can be re-prefilled. Eviction is a cost, recorded in
     `Budget.recompute_tokens`, not a loss.

Then `benchmarks/cross_check_cache.py` per SPEC §1.3, `experiments/tree_engine.py`
per docs/EXPERIMENTS.md#f1, and `examples/01_trace_tree.py`.

Write every test in the phase-1 table of docs/BUILD_PLAN.md.

Also validate the FLOP model from phase 0 against measured throughput, as F1
specifies, and report the ratio. Every FLOP axis in the repository inherits
that number, so if it is far from 1, say so loudly rather than filing it.

Fill in `[[N5a]]`, `[[N5b]]` and `[[N5c]]` in README.md from the full
(non-`--quick`) run: the speedup at branching factor 4 and depth 8, the peak
memory against the independent-sequence baseline, and the maximum logit
deviation.

Gate:

```bash
pytest tests/ -q -m "not slow and not network"
python benchmarks/cross_check_cache.py
PYTHONPATH=experiments python experiments/tree_engine.py --quick
```

The cross-check must report a maximum absolute logit deviation under 1e-3 and
write `benchmarks/results/cross_check_cache.json`.

Report: the achieved deviation and whether it is flat or rising with depth
(rising means positions are drifting — do not proceed); the speedup against
re-prefill at each branching factor and depth; peak memory against the
independent-sequence baseline; the measured block waste; exactness across an
eviction-and-recompute cycle; and the analytic-to-measured FLOP ratio.
