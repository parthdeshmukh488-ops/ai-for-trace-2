# The build transcript

Eleven files, one per phase. Each is written to be pasted into a terminal
coding agent as-is, at the repository root, in order.

## How to use them

1. Open a session at the repository root.
2. Paste the contents of `phase-00.md`.
3. When the agent reports the gate is green, verify it yourself by running the
   gate command.
4. Commit. Then paste the next phase.

Do not paste two phases at once. The gates exist because each layer is the
oracle for the next one, and the ordering matters more here than in most
projects: phase 3 spends real generation budget building a corpus that phases
4, 5 and 8 all read, and a mistake in its sampling conditions is not a bug to
fix later but a corpus to rebuild.

## Before the first paste

The agent should read `AGENT_BRIEF.md` first. The three rules easiest to lose
track of over a long build:

* no number in any file that was not measured by a run that produced a
  `results/*.json`;
* every `V*` carries its model, temperature, sampling parameters and `N` — it
  is a property of a prefix *under a policy*, not of the prefix;
* compute comparisons count the reward model's own forward passes, because
  that is where the usual comparisons go wrong.

## The expensive phases

Phases 3, 7 and 8 generate tens of millions of tokens between them.
`docs/ENGINEERING.md#compute-tiers` has the arithmetic. Two things worth
deciding before starting rather than during:

* whether the fast inference path is available, which changes phase 7 from
  hours to days;
* the sampling temperature, which is baked into the corpus and cannot be
  changed afterwards without regenerating it.

## If a phase goes sideways

Each file ends with a **Report** section listing what to say on completion. If
the agent cannot reach the gate, the useful response is that report with the
blocker named, not a partial implementation presented as finished.

## Local convenience

The brief is tracked as `AGENT_BRIEF.md` so it stays out of tool-specific
naming. If your agent auto-loads a context file from the repository root, copy
it to whatever name that tool expects and leave the copy untracked:

```bash
cp AGENT_BRIEF.md CONTEXT.md && echo "CONTEXT.md" >> .git/info/exclude
```
