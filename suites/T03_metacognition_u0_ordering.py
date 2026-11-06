"""T03: in an independently built shared-budget metacognition world, does the three-term
ISRM utility U0 score below the fixed cyclic schedule, is U_PLUS equivalent to the
conventional MPC that shares its forecaster, and is U0 decision-identical to its
weighted-sum twin?

World and controllers: isrm_sandbox/environments/metacog_budget.py and
isrm_sandbox/controllers/metacog_policies.py, written from the original benchmark's
written contract. Every tuned planner (U0 simplex, MPC lambda_B, U_PLUS coefficients)
is calibrated on development blocks only and with the same budget: every configuration
of every family is scored on the same 12 development episodes (3 blocks per primary cell,
paired across configurations and families) and the family's best mean J wins. The MPC
lambda_B grid is exactly the beta = gamma = 0 slice of the U_PLUS grid, so a U_PLUS
calibration that sets beta = gamma = 0 selects exactly the MPC winner and the U_PLUS - MPC
contrast measures the S and C terms and nothing else. (The two quick-mode ledger records
preceding the full run used successive halving and then non-nested grids; both times U_PLUS
collapsed to an MPC member with a different lambda_B than MPC's own winner, which is why
the grids were nested before the full run. The primary contrast was never touched.) The recalibration of the shared forecaster's P(correct) and
the nonlinear tolerance ratio are also fitted on development blocks.

Replicate i = one base-seed block: every arm runs the block in the four primary
scarce-coupled cells ({stable, changing} x {linear, nonlinear}); the metric is the
equal-weight mean J over those cells. Each arm also runs the ample-coupled stable linear
cell so the ample-minus-scarce difference can be read per arm (extras): the original
reported that U0 does *worse* when the budget is ample, the receding-horizon
procrastination mechanism.
"""
from __future__ import annotations

import numpy as np

from isrm_sandbox.controllers import metacog_policies as MP
from isrm_sandbox.environments import metacog_budget as MB

N_DEV_BLOCKS = 12
N_BEHAVIOUR_BLOCKS = 8
AMPLE_CELL = "ample-coupled-stable-linear"


def dev_cell(j: int) -> str:
    return MB.PRIMARY_CELLS[j % len(MB.PRIMARY_CELLS)]


def collect_pairs(block, cell, policy, cfg, r_nl, fc) -> list:
    """Run a behaviour policy and collect (P(correct), realised correctness) pairs."""
    ep = MB.Episode(block, cell, cfg, r_nl)
    policy.reset(ep)
    pairs = []
    while ep.t < ep.T:
        pairs.extend(MB.hold_forecast_pairs(ep, fc))
        ep.execute(policy.decide(ep, fc))
    return pairs


def calibrate(configs, make, dev_blocks, cfg, r_nl, fc):
    """Score every configuration on the same development episodes; returns (winner, record)."""
    table = []
    for k, params in enumerate(configs):
        js = [MB.run_episode(dev_blocks[j], dev_cell(j), make(params), cfg, r_nl, fc)["J"] for j in range(len(dev_blocks))]
        table.append((-float(np.mean(js)), k))
    table.sort()
    best, runner = table[0][1], (table[1][1] if len(table) > 1 else None)
    record = {
        "config": configs[best],
        "dev_J": round(-table[0][0], 5),
        "n_dev_episodes": len(dev_blocks),
        "runner_up": configs[runner] if runner is not None else None,
        "runner_up_dev_J": round(-table[1][0], 5) if runner is not None else None,
        "family_dev_J_range": [round(-table[-1][0], 5), round(-table[0][0], 5)],
    }
    return configs[best], record


def build(spec, seed_for):
    env = spec.environment
    b = env["budget"]
    cfg = MB.MetacogConfig(
        n_tasks=int(env["tasks"]),
        ticks=int(env["ticks"]),
        horizon=int(env["horizon"]),
        B0=float(b["B0"]),
        Bmax=float(b["Bmax"]),
        refill_scarce=float(b["refill_scarce"]),
        refill_ample=float(b["refill_ample"]),
    )
    dev_blocks = [MB.make_block(np.random.default_rng(seed_for("dev", j)), cfg) for j in range(N_DEV_BLOCKS)]
    r_nl = MB.nonlinear_scale_ratio(dev_blocks[:N_BEHAVIOUR_BLOCKS], cfg)
    fc0 = MB.Forecaster()
    pairs: list = []
    for j in range(N_BEHAVIOUR_BLOCKS):
        pairs += collect_pairs(dev_blocks[j], dev_cell(j), MP.FixedFair(), cfg, r_nl, fc0)
        rnd = MP.RandomFeasible(np.random.default_rng(seed_for("dev_behaviour", j)))
        pairs += collect_pairs(dev_blocks[j], dev_cell(j), rnd, cfg, r_nl, fc0)
    c0, c1 = MB.fit_recalibration(pairs)
    fc = MB.Forecaster(c0, c1)

    u0_cfg, u0_rec = calibrate(MP.u0_simplex(6), lambda p: MP.Planner("U0", **p), dev_blocks, cfg, r_nl, fc)
    mpc_cfg, mpc_rec = calibrate(MP.MPC_GRID, lambda p: MP.Planner("CONVENTIONAL_MPC", **p), dev_blocks, cfg, r_nl, fc)
    up_cfg, up_rec = calibrate(MP.U_PLUS_GRID, lambda p: MP.Planner("U_PLUS", **p), dev_blocks, cfg, r_nl, fc)
    twin = {"w_E": u0_cfg["alpha"], "w_S": u0_cfg["beta"], "w_C": u0_cfg["gamma"]}

    arms = {
        "isrm_u0": lambda: MP.Planner("U0", **u0_cfg),
        "fixed_fair": MP.FixedFair,
        "conventional_mpc": lambda: MP.Planner("CONVENTIONAL_MPC", **mpc_cfg),
        "u_plus": lambda: MP.Planner("U_PLUS", **up_cfg),
        "linear_twin": lambda: MP.Planner("LINEAR_EQUIVALENT", **twin),
        "hold_all": MP.HoldAll,
    }
    selection = {
        "isrm_u0": u0_rec,
        "conventional_mpc": mpc_rec,
        "u_plus": up_rec,
        "linear_twin": twin,
        "recalibration": {"c0": round(c0, 5), "c1": round(c1, 5), "n_pairs": len(pairs)},
        "r_nl": round(r_nl, 5),
        "dev_blocks": N_DEV_BLOCKS,
    }
    return cfg, r_nl, fc, arms, selection


def run(spec, replicates, seed_for):
    cfg, r_nl, fc, arms, selection = build(spec, seed_for)
    wanted = {a.name for a in spec.arms}
    arms = {k: v for k, v in arms.items() if k in wanted}
    results = {k: [] for k in arms}
    per_arm = lambda: {k: [] for k in arms}  # noqa: E731
    extras = {
        "missing_rate": per_arm(),
        "ops_paid_per_episode": per_arm(),
        "J_scarce_coupled_stable_linear": per_arm(),
        "J_ample_coupled_stable_linear": per_arm(),
        "ample_minus_scarce_J": per_arm(),
        "J_linear_cells": per_arm(),
        "J_nonlinear_cells": per_arm(),
        "u0_twin_mismatch_ticks": {k: [] for k in ("isrm_u0", "linear_twin") if k in arms},
    }
    for i in range(replicates):
        block = MB.make_block(np.random.default_rng(seed_for("env", i)), cfg)
        logs = {}
        for name, make in arms.items():
            Js, miss, ops = [], [], []
            for cell in MB.PRIMARY_CELLS:
                r = MB.run_episode(block, cell, make(), cfg, r_nl, fc)
                Js.append(r["J"])
                miss.append(r["missing_rate"])
                ops.append(r["ops_paid"])
                if cell == MB.PRIMARY_CELLS[0]:
                    logs[name] = r["log"]
            amp = MB.run_episode(block, AMPLE_CELL, make(), cfg, r_nl, fc)
            results[name].append(float(np.mean(Js)))
            extras["missing_rate"][name].append(float(np.mean(miss)))
            extras["ops_paid_per_episode"][name].append(float(np.mean(ops)))
            extras["J_scarce_coupled_stable_linear"][name].append(Js[0])
            extras["J_ample_coupled_stable_linear"][name].append(amp["J"])
            extras["ample_minus_scarce_J"][name].append(amp["J"] - Js[0])
            extras["J_linear_cells"][name].append(float(np.mean([Js[0], Js[1]])))
            extras["J_nonlinear_cells"][name].append(float(np.mean([Js[2], Js[3]])))
        if "isrm_u0" in logs and "linear_twin" in logs:
            mism = sum(a != b for a, b in zip(logs["isrm_u0"], logs["linear_twin"]))
            for k in extras["u0_twin_mismatch_ticks"]:
                extras["u0_twin_mismatch_ticks"][k].append(float(mism))
    extras["selection"] = selection
    return results, extras
