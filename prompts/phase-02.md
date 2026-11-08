Read AGENT_BRIEF.md. Phase 1's gate must be green, including the cache
cross-check. This is phase 2 of docs/BUILD_PLAN.md: verification.

Every label in this repository inherits the answer checker's error rate, so
this phase is not plumbing — it sets a floor under every calibration number in
phase 4.

Implement SPEC §2:

1. `verify/answer.py` — the four-stage checker, with the stage that decided
   recorded in the result. Report which extraction rule fired; the fraction of
   traces where extraction fell through to the last-number rule is a quality
   signal for the whole corpus.
2. `verify/sandbox.py` — a fresh subprocess with `RLIMIT_CPU`, `RLIMIT_AS`,
   `RLIMIT_NOFILE`, `RLIMIT_NPROC` and `RLIMIT_FSIZE` set before exec, a
   parent-enforced wall-clock timeout, an empty environment, and a scratch
   directory that is removed afterwards.

   State the limits honestly in the module docstring: resource limits stop
   runaway programs, not hostile ones, and a pure-Python sandbox cannot
   prevent network access or reads outside the scratch directory without
   OS-level support. Say that untrusted code belongs in a container, and that
   this module exists to survive generated code that loops forever or
   allocates 40 GB — which is what generated code actually does.

   Never execute generated code in the host process, not even for a quick
   check during development.
3. `verify/audit.py` — false-accept and false-reject rates with Wilson
   intervals, and the stratified sampling procedure recorded in the report.

Write the phase-2 tests from docs/BUILD_PLAN.md. Each sandbox test must assert
*which* limit stopped the program, not merely that it was stopped — a test
satisfied by the wall-clock timeout is not testing the memory limit.

Then run `experiments/verifier_audit.py` per docs/EXPERIMENTS.md#f3. The hand
labelling is real work and cannot be skipped or automated; 200 samples,
stratified 100 accepted and 100 rejected. Record the count of cases where the
correct label was genuinely unclear as its own category rather than forcing
them into one of the two buckets.

Fill in `[[N5d]]` and `[[N5e]]` in README.md with the measured false-accept
and false-reject rates, each with its interval.

Gate:

```bash
pytest tests/test_answer.py tests/test_sandbox.py -q
PYTHONPATH=experiments python experiments/verifier_audit.py
```

Report: false-accept and false-reject rates with intervals, broken down by
answer type and extraction rule; the symbolic-timeout hit rate; the unclear-
label count; and the sandbox limits you set and how you chose each. Write the
false-accept rate somewhere phase 4 will find it, because it is quoted beside
every ECE.
