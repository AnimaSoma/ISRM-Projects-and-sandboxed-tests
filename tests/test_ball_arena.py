"""Protocol tests for the ball-avoidance arena and its governors (T04)."""
from __future__ import annotations

import numpy as np
import pytest

from isrm_sandbox.controllers import arena_governors as AG
from isrm_sandbox.environments import ball_arena as BA

pytestmark = pytest.mark.protocol

DEFAULT = BA.Cell(12, 100.0, 0.15, 100.0)


@pytest.fixture(scope="module")
def default_worlds():
    return [BA.make_world(np.random.default_rng(1000 + k), DEFAULT) for k in range(3)]


def test_world_is_deterministic_contained_and_speed_preserving():
    a = BA.make_world(np.random.default_rng(5), BA.Cell(12, 100.0, 0.9, 100.0))
    b = BA.make_world(np.random.default_rng(5), BA.Cell(12, 100.0, 0.9, 100.0))
    assert np.array_equal(a["H"], b["H"]) and np.array_equal(a["r"], b["r"])
    H, r = a["H"], a["r"]
    assert H.shape == (BA.TICKS + 1, 12, 4)
    assert np.all(np.hypot(H[:, :, 0], H[:, :, 1]) <= BA.PHYS["radius"] - r + 1e-6)
    speed = np.hypot(H[:, :, 2], H[:, :, 3])
    assert np.allclose(speed, speed[0], rtol=1e-9)  # turns and reflections preserve speed
    assert np.all((speed[0] >= 72.0) & (speed[0] <= 128.0))
    cells = BA.all_cells()
    assert len(cells) == 72 and len(set(cells)) == 72


def test_agent_integration_is_acceleration_limited_and_wall_clamped():
    x, y, vx, vy = BA.integrate_agent(0.0, 0.0, 0.0, 0.0, 115.0, 0.0)
    assert abs(vx - BA.PHYS["accel"] * BA.DT) < 1e-12 and vy == 0.0 and abs(x - vx * BA.DT) < 1e-12
    x, y, vx, vy = BA.integrate_agent(270.9, 0.0, 115.0, 0.0, 115.0, 0.0)
    assert abs(np.hypot(x, y) - (BA.PHYS["radius"] - BA.PHYS["agent_radius"])) < 1e-9 and vx <= 1e-12


class _Stoic:
    """Stays still, senses nothing: the passive reference."""

    def reset(self, cell):
        return None

    def decide(self, tick, hazards, radii, agent):
        return 0.0, 0.0

    def diagnostics(self):
        return {}


def test_energy_accounting_reconciles(default_worlds):
    w = default_worlds[0]
    r = BA.run_agent(w, _Stoic())
    assert abs(r["movement_cost"]) < 1e-12
    expected = DEFAULT.energy - r["collision_cost"] - BA.PHYS["basal"] * r["survival"]
    if r["alive"]:
        assert abs(r["terminal_energy"] - expected) < 1e-6
    for arm in ("isrm_full", "adaptive_energy_only", "fixed_energy_only"):
        r = BA.run_agent(w, AG.make_controller(arm))
        paid = (r["movement_cost"] + r["collision_cost"] + BA.PHYS["basal"] * r["survival"] + r["observation_cost"] + r["model_cost"]
                + r["planning_cost"] + r["governance_cost"] + r["recalibration_cost"])
        if r["alive"]:
            assert abs(DEFAULT.energy - paid - r["terminal_energy"]) < 1e-6, arm


def test_controllers_are_deterministic_given_the_world(default_worlds):
    w = default_worlds[1]
    for arm in ("isrm_full", "adaptive_energy_dwell", "fixed_energy_only"):
        a = BA.run_agent(w, AG.make_controller(arm))
        b = BA.run_agent(w, AG.make_controller(arm))
        assert a == b, arm


def test_isrm_with_zero_extra_weights_equals_adaptive_energy_only(default_worlds):
    for w in default_worlds[:2]:
        zero = BA.run_agent(w, AG.ArenaController("isrm", weights=(0.4, 0.0, 0.0)))
        energy = BA.run_agent(w, AG.make_controller("adaptive_energy_only"))
        for key in ("survival", "terminal_energy", "switches", "recalibrations", "hits"):
            assert zero[key] == energy[key], key


def test_dwell_only_comparator_switches_far_less_than_plain_adaptive(default_worlds):
    plain, dwell, isrm = [], [], []
    for w in default_worlds:
        plain.append(BA.run_agent(w, AG.make_controller("adaptive_energy_only"))["switches"])
        dwell.append(BA.run_agent(w, AG.make_controller("adaptive_energy_dwell"))["switches"])
        isrm.append(BA.run_agent(w, AG.make_controller("isrm_full"))["switches"])
    assert np.mean(plain) >= 3, plain  # the plain adaptive governor does churn models
    assert np.mean(dwell) <= 0.5 * np.mean(plain), (plain, dwell)
    assert np.mean(isrm) <= 0.5 * np.mean(plain), (plain, isrm)


def test_no_switch_ablation_never_leaves_the_inertial_family(default_worlds):
    r = BA.run_agent(default_worlds[2], AG.make_controller("isrm_no_switching"))
    assert r["switches"] == 0 and r["recalibrations"] == 0 and r["turn_fraction"] == 0.0
    assert r["assessments"] > 0  # it still pays for the observer and assessments


def test_dwell_margin_is_isrms_identity_penalty_in_energy_units():
    c = AG.ArenaController("energy_dwell")
    assert abs(c.margin - 0.25 * 0.1 / 0.4) < 1e-12
    assert AG.ArenaController("energy").margin == 0.0 and AG.ArenaController("isrm").margin == 0.0
