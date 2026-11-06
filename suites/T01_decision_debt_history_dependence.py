"""T01: does decision debt produce history dependence at matched belief and capacity,
and do conventional state-carrying controllers without a debt term produce it too?"""
from __future__ import annotations

import numpy as np

from isrm_sandbox.controllers import isrm_debt as D
from isrm_sandbox.environments import changepoint_task as CT


def _arms(ctrl_cfg: dict) -> dict:
    base = dict(ctrl_cfg)
    return {
        "isrm_debt": (D.DebtParams(**base, rho=0.10, lambda_eps=0.5), None),
        "matched_no_debt": (D.DebtParams(**base, rho=0.10, lambda_eps=0.0), None),
        "adaptive_hazard_no_debt": (D.DebtParams(**base, rho=0.10, lambda_eps=0.0, adaptive_hazard=True), None),
        "leaky_accumulator_no_debt": (D.DebtParams(**base, rho=0.10, lambda_eps=0.0, leaky_accumulator=True, lambda_acc=0.5, acc_rho=0.10), None),
        "isrm_debt_lambda1": (D.DebtParams(**base, rho=0.10, lambda_eps=1.0), None),
        "isrm_debt_rho005": (D.DebtParams(**base, rho=0.05, lambda_eps=0.5), None),
        "isrm_debt_rho080": (D.DebtParams(**base, rho=0.80, lambda_eps=0.5), None),
        "isrm_debt_2episodes": (D.DebtParams(**base, rho=0.10, lambda_eps=0.5), 2),
    }


def run(spec, replicates, seed_for):
    env = spec.environment
    task = CT.TaskConfig(**env["task"])
    caps = env["capacities"]
    arms = _arms(env["controller"])
    wanted = {a.name for a in spec.arms}
    results = {name: [] for name in arms if name in wanted}
    extras = {
        "escalation_resolved": {name: [] for name in results},
        "escalation_deferred": {name: [] for name in results},
        "censored_resolved": {name: [] for name in results},
        "censored_deferred": {name: [] for name in results},
        "carried_debt_deferred": {name: [] for name in results},
        "carried_debt_resolved": {name: [] for name in results},
    }
    for i in range(replicates):
        participant = CT.make_participant(np.random.default_rng(seed_for("env", i)), task)
        for name, (params, n_prep) in arms.items():
            if name not in results:
                continue
            r = D.history_effect(params, participant, caps["resolved"], caps["deferred"], prep_episodes=n_prep)
            results[name].append(float(r["effect"]))
            extras["escalation_resolved"][name].append(r["resolved"]["escalation"])
            extras["escalation_deferred"][name].append(r["deferred"]["escalation"])
            extras["censored_resolved"][name].append(float(r["resolved"]["censored"]))
            extras["censored_deferred"][name].append(float(r["deferred"]["censored"]))
            extras["carried_debt_deferred"][name].append(r["deferred"]["eps"])
            extras["carried_debt_resolved"][name].append(r["resolved"]["eps"])
    return results, extras
