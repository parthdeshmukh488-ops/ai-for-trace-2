# Engineering notes

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,verify,plot,data]"
```

On a CPU-only machine install the CPU torch wheels first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The optional fast inference engine is a GPU-only dependency and is genuinely
optional: everything runs without it, more slowly, and F1 records its
throughput as a reference point rather than depending on it.

```bash
pip install -e ".[fast]"     # GPU only
```

Python 3.10 is the floor; CI covers through 3.13.

## Determinism

The guarantee is: **same seed, same machine, same result.** Across machines and
across CPU/GPU it is `allclose`, and the docs should not claim more.

Sampling is the hard part. Rules that make it hold:

* Library code never touches global RNG. Every function that samples takes a
  `generator` or a `seed`.
* Sampling in a batch must not depend on the batch's composition. Drawing
  tokens for a batch of 8 and for the same 8 sequences split into two batches
  of 4 must give the same tokens, or the tree's exactness tests pass while the
  search results silently depend on scheduling. Achieve it by giving each
  sequence its own generator seeded from `(seed, node_id)`, not by drawing
  from one generator across the batch. This is the single most important
  determinism decision in the project and it has to be made at the start,
  because retrofitting it means re-running every experiment.
* The bootstrap, the rollout allocator and every shuffle take generators
  explicitly.
* `tests/test_determinism.py` runs each experiment's `--quick` path twice in
  one process and asserts the JSON payloads are equal, after dropping
  `metadata.wall_seconds` and `metadata.budget.wall_seconds`.

## Compute tiers

| tier | hardware | what it is for |
|---|---|---|
| `smoke` | any CPU, no network | `--quick` paths against the tiny test model; what CI runs |
| `cheap` | one 16 GB accelerator, 0.5B model | the full pipeline end to end, with weaker absolute numbers |
| `main` | one 24-40 GB accelerator, 1.5B model | the numbers that go in the README |

Estimates, to be replaced with measured wall times. Token counts are derived
in docs/EXPERIMENTS.md; these convert them at 2500-4000 tokens/second on the
fast path and 400-800 on the project's own batched decode.

| experiment | tokens | main tier, fast path | main tier, own decode |
|---|---|---|---|
| F1 tree engine | small | minutes | minutes |
| F2 `V*` corpus | ~12M | ~1 hour | ~6 hours |
| F3 verifier audit | reuses F2 | hand labelling | hand labelling |
| F4 PRM calibration | training only | ~2 hours | ~2 hours |
| F5 step localisation | ~4M | ~30 min | ~2.5 hours |
| F6 matched compute | ~90M | ~8 hours | days — reduce scope |
| F7 GRPO | ~1.2M per arm + ~5M eval | ~4 hours | ~1 day |
| F8 selective prediction | ~1.5M | ~15 min | ~1 hour |
| F9 contamination | negligible | minutes | minutes |

F6 is the experiment that decides whether a GPU is enough. Without the fast
inference path, reduce to 150 problems per split and say so in `scope`; do not
drop budgets or methods, because the shape of the frontier is the result and a
frontier with three points is not one.

The `V*` corpus is built **once** and read by F4, F5 and F7. Budget the
generation accordingly: a mistake in F2's sampling conditions costs the whole
corpus, which is why phase 3's coverage validation comes before the generation
rather than after it.

## Memory

* The block cache dominates. For a 1.5B model with 28 layers, 2 KV heads and
  head dimension 128, one token of KV in float16 is
  `28 * 2 * 2 * 128 * 2 bytes = 28 KB`. A tree of 64 nodes averaging 60 tokens
  each is therefore about 100 MB — small, which is the point of sharing. The
  same tree as 64 independent sequences, each re-prefilling a 400-token
  prompt, is several times that.
* Set `max_blocks` from the measured per-token figure and the available memory,
  not from a guess, and have `BlockCache` raise a named error when it cannot
  allocate rather than letting the allocator fail.
* Reward-model scoring of every step of every branch is the other memory
  cliff. Score in batches with an explicit cap.

## Data

* Model ids **and revisions** go in every results JSON. An upstream model
  update silently changing a number is otherwise indistinguishable from a real
  change.
* Dataset name, split and row range go in every results JSON.
* The `V*` corpus manifest records sampling conditions, verifier version and
  the coverage widening factor. A corpus without its conditions is unusable
  and should be deleted rather than kept.
* Nothing downloaded or generated is committed: `weights/`, `adapters/`,
  `rollouts/`, `traces/` and `sandbox_tmp/` are gitignored.

## Performance

In order of how much they matter:

1. **Batch the frontier, not the node.** Expanding one tree node at a time
   leaves the accelerator idle. Gather every node that needs expanding across
   every tree in flight and run them as one batch. This is the difference
   between F6 taking hours and taking days.
2. **Continuous batching over trees.** Trees finish at different times; refill
   the batch as they do rather than waiting for the slowest.
3. **Share the "before" evaluation.** In F7, every arm's pre-training pass@k
   is the same measurement; compute it once.
4. Keep rollouts in `bfloat16` for generation and `float32` for reward-model
   training. Do not train in `bfloat16` without measuring what it does to the
   calibration numbers, which are smaller than its precision in the region
   that matters.
5. `torch.inference_mode()` for all generation and scoring.

## Debugging recipes

**The cache cross-check deviation grows with depth.** Position ids. They must
come from the path length, not the block index. Check a node at depth 2 whose
parent's last block was partially filled — that is where the off-by-one lives.

**A sibling's text appears in another sibling's context.** Copy-on-write is
not firing on the parent's partial last block. `test_partial_last_block_is_
read_only_after_branch` localises it.

**Every search method reports the same accuracy.** The scorer is not being
consulted, or every trace is getting the same score — check the scorer's
output variance before debugging the search.

**A method overspends its token budget.** The budget check is after
generation rather than before the next expansion. Budgets must be checked at
the point of deciding to expand.

**GRPO's reward rises and accuracy does not.** Reward hacking, or a verifier
false-accept the policy has found. Sample the highest-reward completions and
read them; this is the one debugging step that cannot be automated and it is
usually immediately obvious.

**The importance ratio is not 1 on the first inner step.** The behaviour
log-probabilities were recomputed under the updated policy instead of being
cached from the rollout. This makes the first update wrong in a way nothing
else will reveal.

**Entropy collapses in the first 50 steps.** The learning rate, or the
advantage normalisation dividing by a near-zero group standard deviation.
Check the fraction of zero-variance groups first.

## Code style specifics

* `from __future__ import annotations` first, then stdlib, third-party, local.
* `__all__` explicit and sorted, immediately after imports.
* Dataclasses for anything with more than two fields; `frozen=True` unless
  mutation is the point. `ValueEstimate` and `SamplingConditions` are frozen
  by design: a value whose conditions can be edited after the fact is a
  liability.
* Type hints on every public signature; tensor shapes in the docstring.
* No `*args, **kwargs` passthrough in public APIs.
* Errors are specific exception types defined where they are raised, with
  messages naming the offending value and, where relevant, the budget or limit
  that was exceeded.
