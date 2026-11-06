"""Protocol tests for the decision-debt controller re-implementation."""
from __future__ import annotations

import numpy as np
import pytest

from isrm_sandbox.controllers import isrm_debt as D
from isrm_sandbox.environments import changepoint_task as CT

pytestmark = pytest.mark.protocol


def test_debt_is_nonnegative_and_converges_to_alpha_over_rho():
    c = D.DebtController(D.DebtParams(rho=0.1))
    c.q = 1.0  # constant unresolved, maximal discrepancy: alpha = 0.5
    for _ in range(500):
        c.update_debt(settled=False)
        assert c.eps >= 0
    assert abs(c.eps - 0.5 / 0.1) < 1e-6
    c.update_debt(settled=True)  # full settlement capped by available debt
    assert c.eps == pytest.approx(0.0, abs=1e-9)


def test_belief_filter_rises_when_the_model_is_wrong_and_falls_when_right():
    c = D.DebtController(D.DebtParams())
    c.reset_context(model=0)
    for _ in range(15):  # model A wrong: observations mostly 0
        c.update_belief(0)
    assert c.q > 0.9
    c.reset_context(model=0)
    for _ in range(40):  # model A right: observations mostly 1
        c.update_belief(1)
    assert c.q < 0.5


def test_matched_controller_is_history_independent_and_isrm_is_not():
    cfg = CT.TaskConfig()
    rng = np.random.default_rng(7)
    effects_matched, effects_isrm = [], []
    for _ in range(40):
        part = CT.make_participant(rng, cfg)
        effects_matched.append(D.history_effect(D.DebtParams(lambda_eps=0.0), part)["effect"])
        effects_isrm.append(D.history_effect(D.DebtParams(lambda_eps=0.5), part)["effect"])
    assert all(e == 0 for e in effects_matched)
    assert np.mean(effects_isrm) > 0


def test_deferred_preparation_leaves_debt_and_resolved_does_not():
    cfg = CT.TaskConfig()
    part = CT.make_participant(np.random.default_rng(3), cfg)
    r = D.history_effect(D.DebtParams(lambda_eps=0.5), part)
    assert r["deferred"]["eps"] > r["resolved"]["eps"]
    assert r["resolved"]["eps"] < 0.5


def test_revise_is_infeasible_at_low_capacity_and_resolves_episodes_at_full_capacity():
    c = D.DebtController(D.DebtParams())
    c.E_cap = 0.35
    assert D.REVISE not in c.feasible() and D.CHECK in c.feasible()
    cfg = CT.TaskConfig()
    part = CT.make_participant(np.random.default_rng(5), cfg)
    ctrl = D.DebtController(D.DebtParams())
    r = D.run_episode(ctrl, part["prep"][0], capacity=1.0)
    assert r["revisions"] == 1 and ctrl.model == part["prep"][0].rule[0]
    ctrl2 = D.DebtController(D.DebtParams())
    r2 = D.run_episode(ctrl2, part["prep"][0], capacity=0.35)
    assert r2["revisions"] == 0


def test_alternatives_carry_state_across_reset():
    cfg = CT.TaskConfig()
    part = CT.make_participant(np.random.default_rng(9), cfg)
    ah = D.history_effect(D.DebtParams(lambda_eps=0.0, adaptive_hazard=True), part)
    assert ah["deferred"]["h_est"] > ah["resolved"]["h_est"]
    la = D.history_effect(D.DebtParams(lambda_eps=0.0, leaky_accumulator=True), part)
    assert la["deferred"]["acc"] > la["resolved"]["acc"]
