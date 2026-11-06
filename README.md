# ISRM validation sandbox

Independent, isolated tests of the claims made by the ISRM (Interactionist
Self-Regulation Model) projects on this machine. The sandbox exists to answer one
question per claim: **does the effect survive when someone who did not write the
original code re-implements the rule from its written specification and runs it
against matched conventional comparators on fresh worlds?**

Software passing says nothing about whether ISRM helps. The two are reported
separately here, as they are in the original projects.

## Principles

1. **No imports from the original projects.** An import guard refuses any module
   whose file lives outside this folder and its virtual environment. Controllers
   and environments under `isrm_sandbox/` are written from the canonical form,
   the manuscript and the projects' mathematical contracts, not copied.
2. **No network.** Socket creation raises during a run.
3. **Preregistration by construction.** Every test is a frozen JSON spec in
   `specs/` carrying the verbatim source claim, hypotheses, metric, arms, primary
   contrast, direction, pass threshold, alpha and replicate count. The spec is
   content-hashed; a modified spec refuses to load.
4. **Derived seeds.** Seed = SHA-256(spec hash, arm, replicate). Nobody picks seeds.
   Replicate i of every arm sees the same world (paired design).
5. **Matched comparators, always.** Each treatment arm faces at least one
   conventional controller given the same signal, actions and tuning budget
   (CUSUM, Page-Hinkley, EWMA, hysteresis, MPC-style planners), plus an ablation
   that removes the ISRM-specific term.
6. **Verdicts are recorded, not asserted.** `pytest` checks protocol invariants
   only. Scientific verdicts go to the hash-chained ledger
   `reports/ledger.jsonl` with one of these labels:
   `SUPPORTED_IN_SANDBOX`, `POSITIVE_BELOW_PRACTICAL_THRESHOLD`,
   `EQUIVALENT_TO_COMPARATOR_WITHIN_MARGIN`, `NO_ADVANTAGE_DEMONSTRATED`,
   `HARMFUL_IN_TESTED_RANGE`, `INCONCLUSIVE_WIDE_INTERVAL`, `PROTOCOL_FAILURE`.
   A null result is a valid outcome and is kept.

## Install and run

```powershell
cd "D:\ISRM Tests"
$env:UV_CACHE_DIR = "D:\ISRM Tests\.uv-cache"
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe -r requirements.lock
.venv\Scripts\python -m pytest                      # protocol tests (must pass)
.venv\Scripts\python -m isrm_sandbox list           # frozen specs
.venv\Scripts\python -m isrm_sandbox run --quick    # quick-mode verdicts -> ledger
.venv\Scripts\python -m isrm_sandbox run T01_...    # one spec, full replicates
.venv\Scripts\python -m isrm_sandbox verify-ledger  # hash chain intact?
```

## Layout

```
isrm_sandbox/isolation.py      import guard, network kill-switch, derived seeds, hashing
isrm_sandbox/spec.py           TestSpec: freeze / load_frozen with tamper detection
isrm_sandbox/stats.py          paired bootstrap, sign-flip permutation, Holm, verdict rules
isrm_sandbox/ledger.py         append-only hash-chained JSONL ledger
isrm_sandbox/runner.py         CLI; runs suites under isolation, writes the ledger
isrm_sandbox/controllers/      ISRM re-implementations and matched comparators
isrm_sandbox/environments/     independent synthetic worlds (paired by seed)
specs/                         frozen test specifications (one JSON per claim)
suites/<spec id>.py            run(spec, replicates, seed_for) -> {arm: [metric per replicate]}
tests/                         protocol tests (pytest)
reports/                       ledger.jsonl, project ranking, written findings
data/manifests/                SHA-256 manifests for any external data used
```

## Status (2026-11-06, evening)

- `reports/ISRM_PROJECT_RANKING.md`: evidential ranking of every ISRM project on
  the drive, the label-drift finding, the sandbox results and the test plan.
- `specs/`: ten frozen specs (T01, T01b, T02 to T09). T01, T01b, T02, T03, T04 and T08
  have suites and have been run in quick and full mode; T05, T06, T07 and T09 are frozen
  designs awaiting suites (`python -m isrm_sandbox list` shows which). T05 has an
  independent data loader and pre-checks (`reports/T05_data_prechecks.md`).
- New worlds and controllers for the three suites written this evening:
  `environments/metacog_budget.py` + `controllers/metacog_policies.py` (T03, shared-budget
  metacognition: U0, weighted-sum twin, U_PLUS, MPC, FIXED_FAIR, HOLD_ALL),
  `environments/reserve_supply_plant.py` + `controllers/dual_weights.py` (T08, virtual
  queues, matched fixed weights, drift-plus-penalty twin), and
  `environments/ball_arena.py` + `controllers/arena_governors.py` (T04, ball-avoidance
  arena with the ISRM model governor, the dwell-only comparator, adaptive and fixed
  energy-only controllers).
- `tests/`: 81 protocol and formalism tests, all passing (about 3 minutes).
- `reports/ledger.jsonl`: hash-chained verdicts. Quick-mode records precede each full
  record and are kept, including the two T03 quick runs whose comparator grids were
  changed before the full run (explained in the suite docstring), and one protocol
  failure from an early runner bug.

To add a claim: write a `TestSpec` builder in `tools/freeze_specs.py`, run it to
freeze, then write `suites/<id>.py` exposing `run(spec, replicates, seed_for)`.
