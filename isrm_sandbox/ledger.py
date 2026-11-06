"""Append-only, hash-chained results ledger (reports/ledger.jsonl).

Each record carries the hash of the previous record, so deleting or editing an
entry breaks the chain, which `verify_chain` detects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .isolation import sha256_bytes

GENESIS = "0" * 64


def _canonical(record: dict[str, Any]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())
    return (json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip())


def last_hash(path: str | Path) -> str:
    h = GENESIS
    for rec in iter_records(path):
        h = rec["record_hash"]
    return h


def append(path: str | Path, record: dict[str, Any]) -> dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(record)
    rec["prev_hash"] = last_hash(p)
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    rec["record_hash"] = sha256_bytes(_canonical(body))
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
    return rec


def verify_chain(path: str | Path) -> tuple[bool, str]:
    prev = GENESIS
    for i, rec in enumerate(iter_records(path)):
        if rec.get("prev_hash") != prev:
            return False, f"record {i}: prev_hash mismatch"
        body = {k: v for k, v in rec.items() if k != "record_hash"}
        if sha256_bytes(_canonical(body)) != rec.get("record_hash"):
            return False, f"record {i}: record_hash mismatch (record altered)"
        prev = rec["record_hash"]
    return True, "chain intact"
