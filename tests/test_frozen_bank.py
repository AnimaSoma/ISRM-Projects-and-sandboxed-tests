"""Protocol tests for the frozen expert bank world."""
from __future__ import annotations

import numpy as np
import pytest

from isrm_sandbox.environments import frozen_bank as FB

pytestmark = pytest.mark.protocol


@pytest.fixture(scope="module")
def world_and_cfg():
    cfg = FB.FrozenBankConfig(horizon=1024, persistence=80, delay=2)
    return FB.make_world(np.random.default_rng(42), cfg), cfg


def test_world_deterministic():
    cfg = FB.FrozenBankConfig(horizon=128)
    a = FB.make_world(np.random.default_rng(3), cfg)
    b = FB.make_world(np.random.default_rng(3), cfg)
    assert np.array_equal(a["errs"], b["errs"]) and np.array_equal(a["schedule"], b["schedule"])


def test_experts_are_competent_only_on_their_own_regime(world_and_cfg):
    world, cfg = world_and_cfg
    errs, sched = world["errs"], world["schedule"]
    own = np.array([errs[sched[t], t] for t in range(cfg.horizon)])
    assert own.mean() < 0.12, own.mean()  # in-regime experts are good
    assert 0.0 < world["b_ref"] < 0.12
    off = np.array([errs[(sched[t] + 1) % cfg.n_regimes, t] for t in range(cfg.horizon)])
    assert off.mean() > 0.3, off.mean()  # wrong expert is near chance


def test_ordering_oracle_event_never(world_and_cfg):
    world, cfg = world_and_cfg
    oracle = FB.run_episode(world, cfg, FB.OracleScanPolicy(world["schedule"]))
    event = FB.run_episode(world, cfg, FB.CUSUMEventPolicy(theta=0.5, gamma=0.8))
    ewma = FB.run_episode(world, cfg, FB.EWMAScanPolicy(beta=0.95, margin=0.03, max_gap=64))
    dense = FB.run_episode(world, cfg, FB.DenseScanPolicy())
    never = FB.run_episode(world, cfg, FB.NeverScanPolicy())
    assert dense["accuracy"] >= event["accuracy"] - 0.01
    assert oracle["accuracy"] > never["accuracy"] + 0.1
    assert event["accuracy"] > never["accuracy"] + 0.1
    assert ewma["accuracy"] > never["accuracy"] + 0.1
    # dense scans every batch except the last `delay` ones (receipts would never arrive)
    assert never["work"] < event["work"] < dense["work"] <= cfg.n_regimes
    assert dense["work"] > cfg.n_regimes - 0.1
    assert never["scans"] == 1 and oracle["scans"] <= world["regime_changes"] + 1


def test_late_scans_are_suppressed(world_and_cfg):
    world, cfg = world_and_cfg
    dense = FB.run_episode(world, cfg, FB.DenseScanPolicy())
    assert dense["scans"] == cfg.horizon - cfg.delay
