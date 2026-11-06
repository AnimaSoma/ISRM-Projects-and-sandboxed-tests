"""T04: is the Total ISRM v3 survival gain over the adaptive energy-only controller
explained by switching hygiene? The decisive comparator is an energy-only governor given
ISRM's own dwell and a switching margin equal to ISRM's constant coherence-identity
penalty (0.25 x 0.1 in U units, 0.0625 in energy-value units), with no uncertainty or
forecast-disruption measurement.

Arena and controllers: isrm_sandbox/environments/ball_arena.py and
isrm_sandbox/controllers/arena_governors.py, re-created from the original's documented
rules. Nothing is tuned: every arm is a fixed design (the original's frozen weights
0.4 / 0.35 / 0.25 and fee schedule), so no development worlds are needed.

Replicate i = one fresh paired world: the cell is drawn uniformly from the 72-cell
stationary grid (hazards {4, 12, 26} x speed {40, 100, 180} x volatility
{0, .15, .45, .9} x energy {20, 100}) and the whole hazard trajectory is drawn once; every
arm's agent faces the identical balls. Metric: survival seconds (120 s cap). Extras:
terminal energy, hits, model-family switches, paid recalibrations, time in the turn
family, and the paid fee categories.
"""
from __future__ import annotations

import numpy as np

from isrm_sandbox.controllers import arena_governors as AG
from isrm_sandbox.environments import ball_arena as BA

ARMS = ("isrm_full", "adaptive_energy_dwell", "adaptive_energy_only", "fixed_energy_only", "isrm_no_switching")


def run(spec, replicates, seed_for):
    wanted = {a.name for a in spec.arms}
    arms = [a for a in ARMS if a in wanted]
    results = {a: [] for a in arms}
    per_arm = lambda: {a: [] for a in arms}  # noqa: E731
    extras = {
        "terminal_energy": per_arm(),
        "alive_at_120s": per_arm(),
        "hits": per_arm(),
        "switches": per_arm(),
        "recalibrations": per_arm(),
        "assessments": per_arm(),
        "turn_fraction": per_arm(),
        "governance_plus_recalibration_cost": per_arm(),
        "observation_plus_model_cost": per_arm(),
        "movement_cost": per_arm(),
        "hazard_count": per_arm(),
    }
    for i in range(replicates):
        rng = np.random.default_rng(seed_for("env", i))
        world = BA.make_world(rng)
        for a in arms:
            r = BA.run_agent(world, AG.make_controller(a))
            results[a].append(float(r["survival"]))
            extras["terminal_energy"][a].append(float(r["terminal_energy"]))
            extras["alive_at_120s"][a].append(float(r["alive"]))
            extras["hits"][a].append(float(r["hits"]))
            extras["switches"][a].append(float(r["switches"]))
            extras["recalibrations"][a].append(float(r["recalibrations"]))
            extras["assessments"][a].append(float(r["assessments"]))
            extras["turn_fraction"][a].append(float(r["turn_fraction"]))
            extras["governance_plus_recalibration_cost"][a].append(float(r["governance_cost"] + r["recalibration_cost"]))
            extras["observation_plus_model_cost"][a].append(float(r["observation_cost"] + r["model_cost"]))
            extras["movement_cost"][a].append(float(r["movement_cost"]))
            extras["hazard_count"][a].append(float(world["cell"].count))
    extras["selection"] = {
        "adaptive_energy_dwell_margin_energy_units": AG.ArenaController("energy_dwell").margin,
        "isrm_weights": list(AG.ISRM_WEIGHTS),
        "dwell_ticks": AG.MODEL["dwell_ticks"],
        "hysteresis": AG.MODEL["hysteresis"],
        "cells": "uniform draw from the 72-cell stationary grid per replicate",
    }
    return results, extras
