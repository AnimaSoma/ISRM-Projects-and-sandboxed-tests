"""Protocol tests: the isolation guarantees, spec freezing and the ledger chain.

These must always pass. They say nothing about whether ISRM works.
"""
from __future__ import annotations

import json
import socket
import sys

import numpy as np
import pytest

from isrm_sandbox import isolation, ledger, stats
from isrm_sandbox.spec import Arm, TestSpec

pytestmark = pytest.mark.protocol


def _spec(**overrides) -> TestSpec:
    base = dict(
        id="T00_fixture",
        title="fixture",
        source_project="none",
        source_claim="none",
        hypothesis_h1="treatment > comparator",
        hypothesis_h0="no difference",
        metric="score",
        metric_higher_is_better=True,
        primary_contrast=["isrm", "baseline"],
        direction="greater",
        threshold=0.1,
        alpha=0.05,
        n_replicates=8,
        quick_replicates=4,
        arms=[Arm("isrm", "treatment", "x"), Arm("baseline", "comparator", "y")],
        environment={"name": "fixture"},
    )
    base.update(overrides)
    return TestSpec(**base)


def test_network_is_blocked_after_activation():
    isolation.enable_no_network()
    try:
        with pytest.raises(isolation.SandboxViolation):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with pytest.raises(isolation.SandboxViolation):
            socket.getaddrinfo("example.com", 80)
    finally:
        isolation.disable_no_network()


def test_import_guard_blocks_modules_under_forbidden_root(tmp_path, monkeypatch):
    forbidden = tmp_path / "author_project"
    forbidden.mkdir()
    (forbidden / "author_secret_module_xyz.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(forbidden))
    # Allowed roots win over forbidden ones, so the sandbox root is deliberately
    # left out here: the temporary "author project" lives under .pytest_tmp inside it.
    guard = isolation.install_import_guard(
        forbidden_roots=[str(forbidden)],
        allowed_roots=[sys.prefix, sys.base_prefix],
    )
    try:
        assert str(forbidden) not in sys.path, "sys.path should be scrubbed of forbidden roots"
        sys.path.insert(0, str(forbidden))  # simulate a hostile re-insertion
        with pytest.raises(isolation.SandboxViolation):
            __import__("author_secret_module_xyz")
        assert guard.violations and "author_secret_module_xyz" in guard.violations[-1]
        import json as _json  # stdlib still importable

        assert _json is json
    finally:
        isolation.uninstall_import_guard()
        sys.path[:] = [p for p in sys.path if p != str(forbidden)]


def test_seed_derivation_is_stable_and_distinct():
    a = isolation.derive_seed("abc", "isrm", 0)
    b = isolation.derive_seed("abc", "isrm", 0)
    c = isolation.derive_seed("abc", "isrm", 1)
    d = isolation.derive_seed("abc", "baseline", 0)
    assert a == b and a != c and a != d
    assert 0 <= a < 2**63


def test_spec_freeze_roundtrip_and_tamper_detection(tmp_path):
    spec = _spec()
    path = tmp_path / "T00_fixture.json"
    h = spec.freeze(path)
    loaded = TestSpec.load_frozen(path)
    assert loaded.frozen_hash == h and loaded.id == "T00_fixture"
    d = json.loads(path.read_text())
    d["threshold"] = 0.0  # loosen the pass criterion after freezing
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="modified"):
        TestSpec.load_frozen(path)


def test_spec_validation_rejects_bad_contrast():
    with pytest.raises(ValueError):
        _spec(primary_contrast=["isrm", "nope"]).validate()
    with pytest.raises(ValueError):
        _spec(direction="equivalence", equivalence_margin=None).validate()


def test_ledger_chain_detects_edits(tmp_ledger):
    ledger.append(tmp_ledger, {"spec_id": "A", "verdict": "X"})
    ledger.append(tmp_ledger, {"spec_id": "B", "verdict": "Y"})
    ok, _ = ledger.verify_chain(tmp_ledger)
    assert ok
    lines = tmp_ledger.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["verdict"] = "SUPPORTED_IN_SANDBOX"
    lines[0] = json.dumps(rec, sort_keys=True)
    tmp_ledger.write_text("\n".join(lines) + "\n")
    ok, msg = ledger.verify_chain(tmp_ledger)
    assert not ok and "altered" in msg


def test_bootstrap_is_reproducible_and_brackets_the_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, 40)
    y = rng.normal(0.0, 1.0, 40)
    r1 = stats.paired_bootstrap(x, y, np.random.default_rng(7))
    r2 = stats.paired_bootstrap(x, y, np.random.default_rng(7))
    assert r1 == r2
    assert r1[1] < r1[0] < r1[2]


def test_verdict_labels():
    keys = ("supported", "below_threshold", "equivalent", "none", "harmful", "inconclusive", "protocol_failure")
    labels = {k: k.upper() for k in keys}

    def c(lo, hi):
        return stats.Contrast("t", "c", 10, 0, 0, (lo + hi) / 2, lo, hi, 0.05, 0.5, 5, 5, 0)

    assert stats.verdict(c(0.2, 0.4), "greater", 0.1, labels) == "SUPPORTED"
    assert stats.verdict(c(0.02, 0.05), "greater", 0.1, labels) == "BELOW_THRESHOLD"
    assert stats.verdict(c(-0.3, -0.1), "greater", 0.1, labels) == "HARMFUL"
    assert stats.verdict(c(-0.02, 0.03), "greater", 0.1, labels, equivalence_margin=0.05) == "EQUIVALENT"
    assert stats.verdict(c(-0.05, 0.08), "greater", 0.1, labels) == "NONE"
    assert stats.verdict(c(-1.0, 1.0), "greater", 0.1, labels) == "INCONCLUSIVE"
    assert stats.verdict(c(-0.01, 0.01), "equivalence", 0.0, labels, equivalence_margin=0.05) == "EQUIVALENT"


def test_lower_is_better_metric_is_oriented_toward_treatment():
    rng = np.random.default_rng(3)
    treat = np.full(12, 1.0) + rng.normal(0, 0.01, 12)  # lower error
    comp = np.full(12, 2.0) + rng.normal(0, 0.01, 12)
    c = stats.contrast("t", "c", treat, comp, rng, 0.05, higher_is_better=False)
    assert c.mean_diff > 0 and c.ci_low > 0 and c.wins == 12
