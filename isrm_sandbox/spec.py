"""Frozen test specifications.

A TestSpec is the sandbox's preregistration. It names the source claim verbatim,
the hypothesis, the metric, the arms, the primary contrast, the direction, the
pass threshold and the replicate count. `freeze` writes it with a content hash;
`load_frozen` refuses a file whose content no longer matches its hash, so a
spec cannot be edited after the fact without the ledger showing a new hash.
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .isolation import sha256_bytes

DEFAULT_LABELS = {
    "supported": "SUPPORTED_IN_SANDBOX",
    "below_threshold": "POSITIVE_BELOW_PRACTICAL_THRESHOLD",
    "equivalent": "EQUIVALENT_TO_COMPARATOR_WITHIN_MARGIN",
    "none": "NO_ADVANTAGE_DEMONSTRATED",
    "harmful": "HARMFUL_IN_TESTED_RANGE",
    "inconclusive": "INCONCLUSIVE_WIDE_INTERVAL",
    "protocol_failure": "PROTOCOL_FAILURE",
}


@dataclass
class Arm:
    name: str
    role: str  # treatment | comparator | ablation | control
    description: str


@dataclass
class InputRef:
    path: str
    sha256: str
    description: str = ""


@dataclass
class TestSpec:
    __test__ = False  # not a pytest test class

    id: str
    title: str
    source_project: str
    source_claim: str
    hypothesis_h1: str
    hypothesis_h0: str
    metric: str
    metric_higher_is_better: bool
    primary_contrast: list[str]  # [treatment_arm, comparator_arm]
    direction: str  # greater | less | equivalence
    threshold: float
    alpha: float
    n_replicates: int
    quick_replicates: int
    arms: list[Arm]
    environment: dict[str, Any]
    equivalence_margin: float | None = None
    secondary_contrasts: list[list[str]] = field(default_factory=list)
    inputs: list[InputRef] = field(default_factory=list)
    seeds_policy: str = "derived:sha256(spec_hash|arm|replicate)"
    status_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LABELS))
    notes: str = ""
    frozen_at: str | None = None
    frozen_hash: str | None = None

    # ------------------------------------------------------------------ hashing
    def to_dict(self, include_hash: bool = True) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        if not include_hash:
            d.pop("frozen_hash", None)
        return d

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(include_hash=False), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()

    def compute_hash(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    def validate(self) -> None:
        names = [a.name for a in self.arms]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.id}: duplicate arm names")
        for arm in self.primary_contrast:
            if arm not in names:
                raise ValueError(f"{self.id}: primary contrast arm {arm!r} not among arms {names}")
        for pair in self.secondary_contrasts:
            for arm in pair:
                if arm not in names:
                    raise ValueError(f"{self.id}: secondary contrast arm {arm!r} not among arms")
        if self.direction not in {"greater", "less", "equivalence"}:
            raise ValueError(f"{self.id}: direction must be greater|less|equivalence")
        if self.direction == "equivalence" and not self.equivalence_margin:
            raise ValueError(f"{self.id}: equivalence tests need equivalence_margin")
        if not (0 < self.alpha < 0.5):
            raise ValueError(f"{self.id}: alpha out of range")
        if self.n_replicates < 2 or self.quick_replicates < 2:
            raise ValueError(f"{self.id}: need at least 2 replicates")
        roles = {a.role for a in self.arms}
        if "treatment" not in roles or "comparator" not in roles:
            raise ValueError(f"{self.id}: arms must include a treatment and a comparator")

    # ------------------------------------------------------------------ io
    def freeze(self, path: str | Path) -> str:
        self.validate()
        self.frozen_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.frozen_hash = self.compute_hash()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return self.frozen_hash

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestSpec":
        d = dict(d)
        d["arms"] = [Arm(**a) for a in d.get("arms", [])]
        d["inputs"] = [InputRef(**i) for i in d.get("inputs", [])]
        return cls(**d)

    @classmethod
    def load_frozen(cls, path: str | Path) -> "TestSpec":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        spec = cls.from_dict(d)
        if not spec.frozen_hash:
            raise ValueError(f"{path}: spec is not frozen")
        actual = spec.compute_hash()
        if actual != spec.frozen_hash:
            raise ValueError(
                f"{path}: frozen spec has been modified (recorded {spec.frozen_hash[:12]}, actual {actual[:12]}). "
                "Re-freeze under a new id or revision instead of editing in place."
            )
        spec.validate()
        return spec


def list_specs(folder: str | Path) -> list[Path]:
    return sorted(Path(folder).glob("*.json"))
