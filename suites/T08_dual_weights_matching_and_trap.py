"""T08: can constant weights be matched to the virtual-queue ("dual weights") controller
at equal E/S/C violation depths, is that controller a textbook drift-plus-penalty rule,
and does reserve-funded feasibility produce the absorbing zero-reserve trap?

Plant and controllers: isrm_sandbox/environments/reserve_supply_plant.py and
isrm_sandbox/controllers/dual_weights.py (written from the original's V3 protocol).

Procedure (the original's primary cell: reserve-funded fixed actions, signed residuals):
1. Training worlds seed_for("dev", 0..3) give ADAPTIVE's mean R2 depth targets; the
   bounded search (1,000 coarse triples + two refinement rounds) fits FIXED's weights to
   them. Nothing about the evaluation worlds enters the fit.
2. Replicate i = fresh world seed_for("env", i), run in R1, R2 and R3 by every arm.
   Metric: mean work per tick in R2 (lower is better; the regime the match was fitted on).
   Extras: work in R1/R3, R2 depths and C_true per arm, the held-out joint-match score
   (worst normalised |FIXED - ADAPTIVE| depth mismatch per replicate), the R1 absorbing
   zero-reserve fraction, and the number of ticks on which the drift-plus-penalty twin
   chose a different action from ADAPTIVE (expected 0).
"""
from __future__ import annotations

import numpy as np

from isrm_sandbox.controllers import dual_weights as DW
from isrm_sandbox.environments import reserve_supply_plant as RP

N_TRAIN = 4
REGIMES = ("R1", "R2", "R3")


def build(spec, seed_for):
    cfg = RP.PlantConfig()
    train = [RP.make_world(np.random.default_rng(seed_for("dev", j)), cfg) for j in range(N_TRAIN)]
    target = np.zeros(3)
    for w in train:
        s = RP.run_plant(w, "R2", DW.VirtualQueueController(), cfg)
        target += np.array([s[m] for m in DW.METERS])
    target /= N_TRAIN
    rec = DW.match_fixed_weights(train, target, regime="R2", cfg=cfg)
    arms = {
        "adaptive_queues": lambda: DW.VirtualQueueController(),
        "fixed_matched": lambda: DW.FixedWeightController(rec["weights"]),
        "no_c": lambda: DW.VirtualQueueController(no_c=True),
        "dpp_twin": lambda: DW.DriftPlusPenalty(),
    }
    selection = {
        "fixed_matched": rec,
        "training_worlds": N_TRAIN,
        "cell": "reserve-funded fixed actions, signed residuals, V = 1",
    }
    return cfg, arms, selection


def run(spec, replicates, seed_for):
    cfg, arms, selection = build(spec, seed_for)
    wanted = {a.name for a in spec.arms}
    arms = {k: v for k, v in arms.items() if k in wanted}
    results = {k: [] for k in arms}
    per_arm = lambda: {k: [] for k in arms}  # noqa: E731
    extras = {
        "work_R1": per_arm(),
        "work_R3": per_arm(),
        "E_depth_R2": per_arm(),
        "S_depth_R2": per_arm(),
        "C_depth_R2": per_arm(),
        "C_true_R2": per_arm(),
        "C_true_R3": per_arm(),
        "absorbing_fraction_R1": per_arm(),
        "zero_reserve_fraction_R1": per_arm(),
        "sample_freq_R2": per_arm(),
        "heldout_match_worst_mismatch_R2": {k: [] for k in ("fixed_matched",) if k in arms},
        "dpp_twin_mismatch_ticks": {k: [] for k in ("adaptive_queues", "dpp_twin") if k in arms},
    }
    for i in range(replicates):
        world = RP.make_world(np.random.default_rng(seed_for("env", i)), cfg)
        runs = {name: {reg: RP.run_plant(world, reg, make(), cfg) for reg in REGIMES} for name, make in arms.items()}
        for name, r in runs.items():
            results[name].append(r["R2"]["work"])
            extras["work_R1"][name].append(r["R1"]["work"])
            extras["work_R3"][name].append(r["R3"]["work"])
            extras["E_depth_R2"][name].append(r["R2"]["E_depth"])
            extras["S_depth_R2"][name].append(r["R2"]["S_depth"])
            extras["C_depth_R2"][name].append(r["R2"]["C_depth"])
            extras["C_true_R2"][name].append(r["R2"]["C_true"])
            extras["C_true_R3"][name].append(r["R3"]["C_true"])
            extras["absorbing_fraction_R1"][name].append(r["R1"]["absorbing_fraction"])
            extras["zero_reserve_fraction_R1"][name].append(r["R1"]["zero_reserve_fraction"])
            extras["sample_freq_R2"][name].append(r["R2"]["action_freq"]["Sample"])
        if "fixed_matched" in runs and "adaptive_queues" in runs:
            ad = np.array([runs["adaptive_queues"]["R2"][m] for m in DW.METERS])
            fx = np.array([runs["fixed_matched"]["R2"][m] for m in DW.METERS])
            worst, _ = DW.match_loss(fx[None, :], ad)
            extras["heldout_match_worst_mismatch_R2"]["fixed_matched"].append(float(worst[0]))
        if "dpp_twin" in runs and "adaptive_queues" in runs:
            mism = sum(int(np.sum(runs["adaptive_queues"][reg]["actions"] != runs["dpp_twin"][reg]["actions"])) for reg in REGIMES)
            for k in extras["dpp_twin_mismatch_ticks"]:
                extras["dpp_twin_mismatch_ticks"][k].append(float(mism))
    extras["selection"] = selection
    return results, extras
