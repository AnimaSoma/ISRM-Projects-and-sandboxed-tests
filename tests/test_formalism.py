"""Formalism checks: exact algebraic properties of the ISRM utility as written.

These need no simulation. Each one is a structural fact that the original
projects' own audits reported; here it is re-derived independently from the
written definitions so the ranking report can cite it as verified.

  U0(a)     = alpha*E_after(a) - beta*(S_after(a) - S_now) - gamma*(C_after(a) - C_now)
  twin(a)   = w_E*E_after(a) - w_S*S_after(a) - w_C*C_after(a)
  U_PLUS(a) = V_H(a) + V_cont(a) + alpha*E_after(a) - beta*dS(a) - gamma*dC(a)
  MPC(a)    = V_H(a) + V_cont(a) + lambda_B*E_after(a)
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.formalism


def u0(alpha, beta, gamma, E_after, S_after, C_after, S_now, C_now):
    return alpha * E_after - beta * (S_after - S_now) - gamma * (C_after - C_now)


def twin(wE, wS, wC, E_after, S_after, C_after):
    return wE * E_after - wS * S_after - wC * C_after


def test_u0_is_weighted_sum_plus_constant_so_argmax_is_identical():
    rng = np.random.default_rng(0)
    mismatches = 0
    for _ in range(2000):
        n = int(rng.integers(2, 40))
        a, b, g = rng.dirichlet([1, 1, 1])
        E, S, C = rng.random(n), rng.random(n), rng.random(n)
        S_now, C_now = rng.random(), rng.random()
        x = u0(a, b, g, E, S, C, S_now, C_now)
        y = twin(a, b, g, E, S, C)
        assert np.allclose(x - y, b * S_now + g * C_now)  # difference is a candidate-independent constant
        mismatches += int(np.argmax(x) != np.argmax(y))
    assert mismatches == 0


def test_energy_level_cannot_rerank_candidates_without_saturation():
    """E_after(a) = E_now - cost(a): E_now is a common additive constant, so shifting
    the budget level changes no decision. The energy channel therefore cannot
    pace spending on its own; only a saturating or gating transform of E can."""
    rng = np.random.default_rng(1)
    changed = 0
    for _ in range(2000):
        n = int(rng.integers(2, 30))
        cost = rng.random(n)
        S, C = rng.random(n), rng.random(n)
        a, b, g = rng.dirichlet([1, 1, 1])
        choices = set()
        for E_now in (0.05, 0.5, 1.0, 20.0, 1200.0):
            choices.add(int(np.argmax(u0(a, b, g, E_now - cost, S, C, 0.3, 0.3))))
        changed += int(len(choices) > 1)
    assert changed == 0


def test_energy_saturation_is_the_only_route_to_rank_changes():
    """With E_after = clip(E_now - cost, 0, 1) the ranking can change only when the
    clip binds differently across candidates: demonstrate one such case exists and
    that without binding there is none."""
    a, b, g = 0.5, 0.25, 0.25
    cost = np.array([0.0, 0.6])
    S, C = np.array([0.5, 0.1]), np.array([0.2, 0.2])
    low = np.argmax(u0(a, b, g, np.clip(0.5 - cost, 0, 1), S, C, 0.3, 0.3))
    high = np.argmax(u0(a, b, g, np.clip(5.0 - cost, 0, 1), S, C, 0.3, 0.3))
    assert low != high  # saturation at the cap removes the cost penalty entirely
    # and with a cap far above every level the choice is again level-independent
    picks = {int(np.argmax(u0(a, b, g, np.clip(E - cost, 0, 100), S, C, 0.3, 0.3))) for E in (2.0, 5.0, 50.0)}
    assert len(picks) == 1


def test_u_plus_with_zero_surprise_and_coherence_weights_is_mpc():
    rng = np.random.default_rng(2)
    for _ in range(500):
        n = int(rng.integers(2, 50))
        V_H, V_cont, E = rng.random(n) * 9, rng.random(n) * 9, rng.random(n)
        dS, dC = rng.normal(size=n), rng.normal(size=n)
        lam = 3.4145
        u_plus = V_H + V_cont + lam * E - 0.0 * dS - 0.0 * dC
        mpc = V_H + V_cont + lam * E
        assert np.array_equal(u_plus, mpc)


def test_hypertuner_gradient_on_penalty_weights_is_never_positive():
    """U = alpha*E - beta*dS - gamma*dC - eps*debt with dC, debt >= 0 gives
    dU/dgamma = -dC <= 0 and dU/deps = -debt <= 0, so gradient *ascent* on the
    penalty weights can only drive them to their lower bounds, whatever the
    telemetry. A 'tuner' built this way is not responding to the GPU."""
    rng = np.random.default_rng(3)
    gamma, eps = 1.4, 0.4
    lo = 0.05
    eta = 0.02
    for _ in range(400):
        dC = abs(rng.normal(0.2, 0.1))
        debt = abs(rng.normal(0.5, 0.2))
        gamma = max(lo, gamma + eta * (-dC))
        eps = max(lo, eps + eta * (-debt))
    assert eps == lo
    assert gamma < 1.4 and gamma >= lo


def test_flock_hybrid_coherence_gate_is_unreachable_at_60hz():
    """Hybrid gate as reconstructed in the flock benchmark: pl <- 0.85 pl + 0.15 ee,
    dC <- 0.6 dC + 0.4 (0.55 pl + 0.45 fe), fe = clip(near/6, 0, 1), with ee the
    chord of the per-tick heading change. At 60 Hz and omega_max ~1.5 rad/s the
    chord is <= 2 sin(0.0125) ~ 0.025, so steady-state dC <= 0.55*0.025 + 0.45 < 0.55
    even with the maximum neighbour count. The RECALIBRATE threshold theta_C = 0.55
    can therefore never fire from this channel."""
    omega_max, hz = 1.5, 60.0
    ee = 2.0 * np.sin(omega_max / hz / 2.0)
    pl = dC = 0.0
    for _ in range(5000):
        pl = min(1.0, 0.85 * pl + 0.15 * ee)
        dC = min(1.0, 0.6 * dC + 0.4 * (0.55 * pl + 0.45 * 1.0))
    assert dC < 0.55, dC
    assert dC > 0.45  # the channel saturates just below the gate


def test_dual_weights_update_is_lyapunov_virtual_queue_drift_plus_penalty():
    """Q_i <- max(0, Q_i + r_i); action = argmin[-V*work(a) + sum_i Q_i * r_hat_i(a)].
    This is the textbook drift-plus-penalty rule (Neely): minimise penalty*V plus
    the queue-weighted drift. Verify the two formulations pick identical actions on
    random instances, so 'dual weights' adds no decision beyond the standard rule."""
    rng = np.random.default_rng(4)
    for _ in range(1000):
        n_act, n_q = int(rng.integers(2, 6)), 3
        Q = np.abs(rng.normal(size=n_q))
        work = rng.random(n_act)
        r_hat = rng.normal(size=(n_act, n_q))
        V = 1.0
        isrm_choice = int(np.argmin(-V * work + r_hat @ Q))
        # Neely form: minimise V * penalty + sum_i Q_i * (arrival_i - service_i); here penalty = -work
        dpp_choice = int(np.argmin(V * (-work) + (r_hat * Q).sum(axis=1)))
        assert isrm_choice == dpp_choice
