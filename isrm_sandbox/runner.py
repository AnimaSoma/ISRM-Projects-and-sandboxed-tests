"""Runner: executes a frozen spec's suite under isolation and records a verdict.

A suite is a module suites/<spec id>.py exposing

    def run(spec: TestSpec, replicates: int, seed_for) -> dict[str, list[float]]

where seed_for(arm_name, replicate) returns the derived seed, and the returned
mapping gives one metric value per replicate for every arm named in the spec.
Replicate i of every arm must share the same environment draw (paired design):
suites obtain it via seed_for("env", i).
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from . import isolation, ledger, stats
from .spec import TestSpec, list_specs

ROOT = isolation.SANDBOX_ROOT
SPECS = ROOT / "specs"
SUITES = ROOT / "suites"
LEDGER = ROOT / "reports" / "ledger.jsonl"


def _seed_factory(spec_hash: str) -> Callable[[str, int], int]:
    def seed_for(arm: str, replicate: int) -> int:
        return isolation.derive_seed(spec_hash, arm, replicate)

    return seed_for


def load_suite(spec_id: str):
    if str(SUITES) not in sys.path:
        sys.path.insert(0, str(SUITES))
    return importlib.import_module(spec_id)


def run_spec(spec_path: Path, quick: bool = False, dry: bool = False) -> dict:
    spec = TestSpec.load_frozen(spec_path)
    fp = isolation.activate()
    inputs = isolation.verify_inputs([i.__dict__ for i in spec.inputs])
    code_hash = isolation.sha256_tree(ROOT)
    n = spec.quick_replicates if quick else spec.n_replicates
    seed_for = _seed_factory(spec.frozen_hash)
    started = time.time()
    record: dict = {
        "spec_id": spec.id,
        "spec_hash": spec.frozen_hash,
        "code_hash": code_hash,
        "mode": "quick" if quick else "full",
        "replicates": n,
        "env": fp,
        "inputs": inputs,
    }
    if dry:
        record["verdict"] = "DRY_RUN"
        record["seconds"] = round(time.time() - started, 3)
        return record
    if not (SUITES / f"{spec.id}.py").exists():
        # A frozen spec whose suite has not been written yet: a plan, not a result.
        record["verdict"] = "SUITE_NOT_IMPLEMENTED"
        record["seconds"] = round(time.time() - started, 3)
        return record
    try:
        suite = load_suite(spec.id)
        out = suite.run(spec, n, seed_for)
        extras: dict = {}
        if isinstance(out, tuple):
            results, extras = out
        else:
            results = out
        missing = {a.name for a in spec.arms} - set(results)
        if missing:
            raise RuntimeError(f"suite returned no results for arms {sorted(missing)}")
        lengths = {len(v) for v in results.values()}
        if lengths != {n}:
            raise RuntimeError(f"every arm must return exactly {n} values, got lengths {lengths}")
        rng = np.random.default_rng(isolation.derive_seed(spec.frozen_hash, "bootstrap"))
        t_arm, c_arm = spec.primary_contrast
        primary = stats.contrast(
            t_arm, c_arm, results[t_arm], results[c_arm], rng, spec.alpha, spec.metric_higher_is_better
        )
        secondaries = [
            stats.contrast(a, b, results[a], results[b], rng, spec.alpha, spec.metric_higher_is_better)
            for a, b in spec.secondary_contrasts
        ]
        label = stats.verdict(primary, spec.direction, spec.threshold, spec.status_labels, spec.equivalence_margin)
        record.update(
            {
                "arm_means": {k: float(np.mean(v)) for k, v in results.items()},
                "arm_sd": {k: float(np.std(v, ddof=1)) for k, v in results.items()},
                "primary": primary.to_dict(),
                "secondary": [s.to_dict() for s in secondaries],
                "verdict": label,
                "extras": {
                    name: {arm: float(np.mean(vals)) for arm, vals in per_arm.items()}
                    for name, per_arm in extras.items()
                    if name != "selection" and isinstance(per_arm, dict)
                },
                "selection": extras.get("selection", {}),
            }
        )
    except Exception as exc:  # a protocol failure is itself a recorded outcome
        record["verdict"] = spec.status_labels["protocol_failure"]
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc(limit=8)
    record["seconds"] = round(time.time() - started, 3)
    ledger.append(LEDGER, record)
    return record


def summarize(record: dict) -> str:
    lines = [
        f"[{record['spec_id']}] verdict: {record['verdict']}  "
        f"(mode={record['mode']}, n={record['replicates']}, {record.get('seconds', 0)} s)"
    ]
    if "primary" in record:
        p = record["primary"]
        lines.append(
            f"  {p['treatment']} - {p['comparator']}: {p['mean_diff']:+.5g}  "
            f"{int(round((1 - p['alpha']) * 100))}% CI [{p['ci_low']:+.5g}, {p['ci_high']:+.5g}]  "
            f"p_sign={p['p_sign_flip']:.4f}  W/L/T={p['wins']}/{p['losses']}/{p['ties']}"
        )
        for k, v in record["arm_means"].items():
            lines.append(f"  mean[{k}] = {v:.6g}  (sd {record['arm_sd'][k]:.3g})")
        for s in record.get("secondary", []):
            lines.append(
                f"  secondary {s['treatment']} - {s['comparator']}: {s['mean_diff']:+.5g} "
                f"[{s['ci_low']:+.5g}, {s['ci_high']:+.5g}]"
            )
        for name, per_arm in record.get("extras", {}).items():
            if name == "selection":
                continue
            lines.append(f"  extra[{name}]: " + ", ".join(f"{k}={v:.4g}" for k, v in per_arm.items()))
        if record.get("selection"):
            lines.append(f"  selection: {record['selection']}")
    if "error" in record:
        lines.append(f"  error: {record['error']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m isrm_sandbox", description="ISRM validation sandbox")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list frozen specs")
    r = sub.add_parser("run", help="run one or more specs under isolation")
    r.add_argument("ids", nargs="*", help="spec ids (default: all)")
    r.add_argument("--quick", action="store_true", help="use quick_replicates")
    r.add_argument("--dry", action="store_true", help="verify spec, inputs and isolation only")
    sub.add_parser("verify-ledger", help="check the hash chain of reports/ledger.jsonl")
    v = sub.add_parser("verify-spec", help="verify a frozen spec's hash")
    v.add_argument("path")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for p in list_specs(SPECS):
            try:
                s = TestSpec.load_frozen(p)
                state = "suite ready" if (SUITES / f"{s.id}.py").exists() else "spec only "
                print(f"{s.id:44s} {s.frozen_hash[:12]}  [{state}]  {s.title}")
            except Exception as exc:
                print(f"{p.name:44s} INVALID: {exc}")
        return 0
    if args.cmd == "verify-ledger":
        ok, msg = ledger.verify_chain(LEDGER)
        print(msg)
        return 0 if ok else 1
    if args.cmd == "verify-spec":
        s = TestSpec.load_frozen(args.path)
        print(f"{s.id} OK {s.frozen_hash}")
        return 0
    if args.cmd == "run":
        paths = list_specs(SPECS)
        if args.ids:
            wanted = set(args.ids)
            paths = [p for p in paths if p.stem in wanted]
            if not paths:
                print("no matching specs", file=sys.stderr)
                return 2
        rc = 0
        for p in paths:
            rec = run_spec(p, quick=args.quick, dry=args.dry)
            print(summarize(rec))
            if rec["verdict"] == "PROTOCOL_FAILURE":
                rc = 1
        return rc
    return 2
