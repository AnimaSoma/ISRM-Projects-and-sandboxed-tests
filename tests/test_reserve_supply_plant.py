"""Protocol tests for the reserve-and-supply plant and the dual-weights controllers (T08)."""
from __future__ import annotations

import numpy as np
import pytest

from isrm_sandbox.controllers import dual_weights as DW
from isrm_sandbox.environments import reserve_supply_plant as RP

pytestmark = pytest.mark.protocol

CFG = RP.PlantConfig()


def _world(seed: int, cfg=CFG) -> dict:
    return RP.make_world(np.random.default_rng(seed), cfg)


def test_world_is_deterministic_with_the_declared_deficit_schedule():
    a, b = _world(3), _world(3)
    assert np.array_equal(a["supply"], b["supply"]) and np.array_equal(a["kappa"]["R2"], b["kappa"]["R2"])
    assert a["onsets"][0] == CFG.first_deficit
    gaps = np.diff(a["onsets"])
    assert np.all((gaps >= CFG.deficit_gap - CFG.deficit_jitter) & (gaps <= CFG.deficit_gap + CFG.deficit_jitter))
    for onset in a["onsets"]:
        assert np.all(a["supply"][onset : onset + CFG.deficit_len] == 0.0)
        assert a["supply"][onset - 1] == CFG.supply
    assert np.all(a["kappa"]["R1"] == CFG.kappa0)
    assert set(np.unique(a["kappa"]["R3"])) <= {CFG.kappa0, *CFG.kappa_targets}
    # R3 changes at onsets, R2 150..250 ticks later
    changes2 = np.flatnonzero(np.diff(a["kappa"]["R2"]) != 0) + 1
    changes3 = np.flatnonzero(np.diff(a["kappa"]["R3"]) != 0) + 1
    assert set(changes3) <= set(a["onsets"])
    for c in changes2:
        lags = c - a["onsets"]
        assert np.any((lags >= CFG.r2_lag[0]) & (lags <= CFG.r2_lag[1]))


def test_feasibility_and_effort_curtailment():
    w = _world(1)
    plant = RP.Plant(w, "R1", CFG)
    plant.R = 0.0
    f = plant.feasible()
    assert not f[RP.SAMPLE] and not f[RP.UPDATE] and f[RP.ACT_FULL]
    plant.t = 10  # supply 0.10, cost 0.15 per unit: Act1 is curtailed to 2/3 and R stays 0
    plant.step(RP.ACT_FULL)
    assert abs(plant.work_hist[10] - 0.10 / 0.15) < 1e-12 and plant.R == 0.0
    plant.R = 0.05
    assert plant.feasible()[RP.SAMPLE] and not plant.feasible()[RP.UPDATE]
    plant.t = 11
    plant.step(RP.SAMPLE)
    assert plant.S == 0.0 and plant.s_known == CFG.supply and abs(plant.R - (0.05 + 0.10 - 0.01)) < 1e-12


def test_update_is_an_oracle_refit_and_resets_c():
    w = _world(2)
    plant = RP.Plant(w, "R3", CFG)
    plant.t = int(w["onsets"][0]) + 5  # kappa_true has changed in R3
    plant.R = 0.5
    plant.C = 0.4
    kt = plant.kappa_true[plant.t]
    plant.step(RP.UPDATE)
    assert plant.kappa_model == kt and plant.C == 0.0
    assert plant.Ctrue_hist[plant.t - 1] == 0.0


def test_adaptive_and_drift_plus_penalty_twin_are_identical_tick_for_tick():
    for seed in (10, 11):
        w = _world(seed)
        for regime in ("R1", "R2", "R3"):
            a = RP.run_plant(w, regime, DW.VirtualQueueController(), CFG)
            d = RP.run_plant(w, regime, DW.DriftPlusPenalty(), CFG)
            assert np.array_equal(a["actions"], d["actions"]), (seed, regime)
            assert a["work"] == d["work"]


def test_no_c_holds_its_consistency_queue_at_zero():
    w = _world(12)
    ctrl = DW.VirtualQueueController(no_c=True)
    plant = RP.Plant(w, "R2", CFG)
    ctrl.reset()
    while plant.t < 600:
        a = ctrl.act(plant)
        ctrl.observe(plant.step(a))
        assert ctrl.Q[2] == 0.0
    assert ctrl.Q[0] >= 0 and ctrl.Q[1] >= 0


def test_batched_fixed_plant_matches_the_scalar_controller():
    w = _world(13)
    weights = np.array([[4.0, 1.3, 3.7], [1.0, 1.0, 1.0], [0.0, 8.0, 0.0], [64.0, 0.25, 2.0]])
    for regime in ("R1", "R2"):
        batch = RP.BatchedFixedPlant(w, regime, weights, cfg=CFG).run()
        for k, wt in enumerate(weights):
            scalar = RP.run_plant(w, regime, DW.FixedWeightController(wt), CFG)
            for key in ("work", "E_depth", "S_depth", "C_depth", "C_true"):
                assert abs(scalar[key] - batch[key][k]) < 1e-9, (regime, k, key)


def test_matching_search_recovers_a_triple_from_its_own_depths():
    worlds = [_world(100 + j) for j in range(2)]
    truth = np.array([[4.0, 1.0, 2.0]])
    target = DW.evaluate_candidates(truth, worlds, "R2", 1.0, False, CFG)[0]
    rec = DW.match_fixed_weights(worlds, target, cfg=CFG)
    assert rec["matched"] and rec["worst_normalised_mismatch"] <= 1e-9
    assert rec["n_candidates"] >= 1000


def test_adaptive_run_summary_fields_and_absorbing_diagnostic():
    w = _world(14)
    r = RP.run_plant(w, "R1", DW.VirtualQueueController(), CFG)
    assert set(r) >= {"work", "E_depth", "S_depth", "C_depth", "C_true", "absorbing_fraction", "absorbing_ticks", "action_freq", "actions"}
    assert 0.0 <= r["absorbing_fraction"] <= 1.0 and 0.0 <= r["work"] <= 1.0
    assert abs(sum(r["action_freq"].values()) - 1.0) < 1e-9
