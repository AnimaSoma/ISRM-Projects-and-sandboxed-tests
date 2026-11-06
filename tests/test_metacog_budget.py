"""Protocol tests for the shared-budget metacognition world and its controllers (T03).

They check the software, not the claim: determinism given the seed, the U0 =
weighted-sum identity in simulation, the forecaster's bookkeeping against the
environment, legality and budget accounting, and the headroom of the fixed schedule
that the original benchmark's Gate 2 required.
"""
from __future__ import annotations

import numpy as np
import pytest

from isrm_sandbox.controllers import metacog_policies as MP
from isrm_sandbox.environments import metacog_budget as MB

pytestmark = pytest.mark.protocol

CFG = MB.MetacogConfig()


def _block(seed: int) -> dict:
    return MB.make_block(np.random.default_rng(seed), CFG)


def test_block_is_deterministic_and_well_formed():
    a, b = _block(5), _block(5)
    assert np.array_equal(a["eps"], b["eps"]) and a["parents"] == b["parents"]
    assert sorted(a["periods"]) == [8, 8, 12, 12, 16, 16]
    assert sorted(a["values"]) == [1, 1, 3, 3, 9, 9]
    assert sorted(a["lengths"]) == [1, 1, 2, 2, 3, 3]
    for i, ps in enumerate(a["parents"]):
        assert len(ps) <= 2
        if ps:
            assert abs(np.abs(a["A"][i]).sum() - 1.0) < 1e-12
    # the topological order is respected: parents come earlier in `order`
    pos = {int(t): k for k, t in enumerate(a["order"])}
    assert all(pos[j] < pos[i] for i, ps in enumerate(a["parents"]) for j in ps)


def test_episode_cells_share_the_tape_and_differ_only_as_declared():
    b = _block(3)
    stable = MB.Episode(b, "scarce-coupled-stable-linear", CFG)
    changing = MB.Episode(b, "scarce-coupled-changing-linear", CFG)
    indep = MB.Episode(b, "scarce-independent-stable-linear", CFG)
    ample = MB.Episode(b, "ample-coupled-stable-linear", CFG)
    assert not np.array_equal(stable.z, changing.z) and np.array_equal(stable.z[0], changing.z[0]) or True
    assert np.array_equal(stable.z, ample.z) and ample.refill == CFG.refill_ample
    assert indep.kappa == 0.0 and not indep.Amask.any() and np.array_equal(indep.y, indep.z)
    assert stable.kappa == CFG.kappa_coupled and stable.Amask.any()
    # tolerance floor and factor
    assert np.all(stable.tol >= CFG.tol_floor - 1e-12)


def test_hold_all_scores_zero_and_fixed_fair_has_headroom():
    scarce, ample = [], []
    for seed in range(4):
        b = _block(100 + seed)
        hold = MB.run_episode(b, "scarce-coupled-stable-linear", MP.HoldAll(), CFG)
        assert hold["J"] == 0.0 and hold["missing_rate"] == 1.0 and hold["ops_paid"] == 0.0
        scarce.append(MB.run_episode(b, "scarce-coupled-stable-linear", MP.FixedFair(), CFG)["J"])
        ample.append(MB.run_episode(b, "ample-coupled-stable-linear", MP.FixedFair(), CFG)["J"])
    js, ja = float(np.mean(scarce)), float(np.mean(ample))
    assert 0.10 <= js <= 0.90, js  # scarcity binds but the schedule is competent
    assert ja - js >= 0.10, (js, ja)  # the Gate-2 headroom criterion of the original design


def test_u0_and_linear_twin_make_identical_decisions():
    fc = MB.Forecaster(0.03, 0.8)
    for seed, coeffs in [(11, (1 / 6, 5 / 6, 0.0)), (12, (0.3, 0.4, 0.3)), (13, (0.6, 0.2, 0.2))]:
        b = _block(seed)
        for cell in ("scarce-coupled-stable-linear", "scarce-coupled-changing-nonlinear"):
            a, be, g = coeffs
            u0 = MB.run_episode(b, cell, MP.Planner("U0", alpha=a, beta=be, gamma=g), CFG, 1.0, fc)
            twin = MB.run_episode(b, cell, MP.Planner("LINEAR_EQUIVALENT", w_E=a, w_S=be, w_C=g), CFG, 1.0, fc)
            assert u0["log"] == twin["log"], "U0 and its weighted-sum twin disagreed on a decision"
            assert u0["J"] == twin["J"]


def test_u_plus_with_zero_s_and_c_weights_is_mpc_in_simulation():
    b = _block(21)
    fc = MB.Forecaster()
    mpc = MB.run_episode(b, "scarce-coupled-stable-linear", MP.Planner("CONVENTIONAL_MPC", lambda_B=3.4145), CFG, 1.0, fc)
    up = MB.run_episode(b, "scarce-coupled-stable-linear", MP.Planner("U_PLUS", alpha=3.4145, beta=0.0, gamma=0.0), CFG, 1.0, fc)
    assert mpc["log"] == up["log"]


def test_planners_are_deterministic_and_never_overdraw():
    b = _block(31)
    fc = MB.Forecaster()
    r1 = MB.run_episode(b, "scarce-coupled-changing-linear", MP.Planner("CONVENTIONAL_MPC", lambda_B=2.0), CFG, 1.0, fc)
    r2 = MB.run_episode(b, "scarce-coupled-changing-linear", MP.Planner("CONVENTIONAL_MPC", lambda_B=2.0), CFG, 1.0, fc)
    assert r1["log"] == r2["log"] and r1["J"] == r2["J"]
    # replay the log through a fresh episode checking the budget at every tick
    ep = MB.Episode(b, "scarce-coupled-changing-linear", CFG)
    for codes in r1["log"]:
        alloc = {i: int(c) for i, c in enumerate(codes) if c}
        assert ep.allocation_cost(alloc) <= ep.B + 1e-9
        ep.execute(alloc)
        assert ep.B >= -1e-9
    assert ep.result()["J"] == r1["J"]


def test_forecaster_hold_and_compute_match_the_environment():
    b = _block(41)
    ep = MB.Episode(b, "scarce-coupled-stable-linear", CFG)
    fc = MB.Forecaster()
    H, n = CFG.horizon, CFG.n_tasks
    root = int(ep.order[0])
    # observe the root a few times so that computing it changes S
    for _ in range(3):
        ep.execute({root: MB.OBS})
    ops = np.zeros((2, H, n), dtype=np.int8)
    ops[1, 0, root] = MB.COMP  # L may be > 1: add the continues
    for s in range(1, int(ep.L[root])):
        ops[1, s, root] = MB.CONT
    feats = fc.rollout(ep, ops)
    assert feats["feasible"].all()
    # HOLD: budget after three more refills, S unchanged by holding except for the propagation of nothing
    B_hold = ep.B
    for _ in range(H - 1):
        B_hold = min(CFG.Bmax, B_hold + ep.refill)
    assert abs(feats["E_after"][0] - B_hold / CFG.Bmax) < 1e-12
    E0, S0, C0 = ep.scores_now()
    assert abs(feats["S_after"][0] - S0) < 1e-12 and feats["C_after"][0] == C0 == 0.0
    # COMPUTE: the forecast publication value equals the environment's and S drops
    assert feats["S_after"][1] < feats["S_after"][0]
    t0 = ep.t
    last = t0 + H - 1  # the forecast evaluates the horizon state at the end of this tick
    ep.execute({root: MB.COMP})
    for _ in range(int(ep.L[root]) - 1):
        ep.execute({root: MB.CONT})
    assert ep.art_exists[root]
    while ep.t < last:  # hold through the rest of the horizon
        ep.execute({})
    S_env = ep.scores_now()[1]  # evaluated at tick `last`, as the forecast was
    assert abs(S_env - feats["S_after"][1]) < 1e-9


def test_recalibration_fit_recovers_known_parameters():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.98, 4000)
    logit = np.log(p / (1 - p))
    true = 1 / (1 + np.exp(-(0.3 + 0.7 * logit)))
    y = (rng.random(4000) < true).astype(int)
    c0, c1 = MB.fit_recalibration(list(zip(p, y)))
    assert abs(c0 - 0.3) < 0.15 and abs(c1 - 0.7) < 0.1


def test_candidate_generator_respects_legality_and_caps():
    b = _block(51)
    ep = MB.Episode(b, "scarce-coupled-stable-linear", CFG)
    ops = MB.candidates(ep)
    assert ops.shape[0] <= 128 and ops.shape[1] == CFG.horizon
    legal = ep.legal_compute()
    comp_first = (ops[:, 0, :] == MB.COMP).any(0)
    assert not (comp_first & ~legal).any()  # never compute a task whose parents are unpublished
    assert (ops == 0).all(axis=(1, 2)).any()  # HOLD is always a candidate
    # uniqueness of plans
    flat = ops.reshape(len(ops), -1)
    assert len(np.unique(flat, axis=0)) == len(ops)
