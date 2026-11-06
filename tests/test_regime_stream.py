"""Protocol tests for the independent regime stream: determinism, sane regime
statistics, and the ordering oracle > adaptive > never so that controller
comparisons run in a world where triggering actually matters.
"""
from __future__ import annotations

import numpy as np
import pytest

from isrm_sandbox.controllers import baselines as B
from isrm_sandbox.environments import regime_stream as RS

pytestmark = pytest.mark.protocol


def test_world_is_deterministic_given_seed():
    cfg = RS.StreamConfig(horizon=64)
    w1 = RS.make_world(np.random.default_rng(5), cfg)
    w2 = RS.make_world(np.random.default_rng(5), cfg)
    assert np.array_equal(w1["X"], w2["X"]) and np.array_equal(w1["schedule"], w2["schedule"])


def test_regime_persistence_is_roughly_as_configured():
    cfg = RS.StreamConfig(horizon=20000, persistence=50, n_regimes=3)
    world = RS.make_world(np.random.default_rng(2), cfg)
    sched = world["schedule"]
    changes = int(np.sum(sched[1:] != sched[:-1]))
    mean_dur = cfg.horizon / (changes + 1)
    assert 40 < mean_dur < 60, mean_dur
    assert np.all(sched[1:][sched[1:] != sched[:-1]] != sched[:-1][sched[1:] != sched[:-1]])


def test_oracle_beats_adaptive_beats_never():
    cfg = RS.StreamConfig(horizon=400, persistence=60, delay=2, dim=8, n_regimes=3)
    bank = RS.BankConfig(n_experts=3)
    world = RS.make_world(np.random.default_rng(9), cfg)

    def run(trig):
        return RS.run_episode(world, cfg, bank, trig, np.random.default_rng(1))

    oracle = run(RS.OracleTrigger(world["schedule"], cfg.delay))
    cusum = run(B.CUSUM(h=0.6, drift=0.05))
    never = run(B.NeverTrigger())
    assert oracle.accuracy > cusum.accuracy > never.accuracy, (oracle.accuracy, cusum.accuracy, never.accuracy)
    assert oracle.triggers == world["regime_changes"]
    assert never.work < cusum.work < oracle.work * 3  # triggering costs work


def test_result_dict_fields():
    cfg = RS.StreamConfig(horizon=32)
    world = RS.make_world(np.random.default_rng(0), cfg)
    r = RS.run_episode(world, cfg, RS.BankConfig(), B.NeverTrigger(), np.random.default_rng(0))
    d = r.to_dict()
    assert set(d) == {"accuracy", "work", "switches", "triggers", "regime_changes", "final_quarter_accuracy"}
