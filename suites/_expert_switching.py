"""Shared implementation for the expert-switching specs (T02 family).

Builds the candidate families, performs comparator selection on development
worlds that are disjoint from evaluation worlds, and evaluates every arm on the
same evaluation worlds (paired design).

Selection rules re-implemented from the original studies' written contract:
  cost       : min work among candidates with acc >= max(ref_acc - 0.5pp, best_acc - 1pp);
               ties -> higher acc, then lower id; fallback highest acc
  budget     : highest acc with work <= ref_work; ties -> lower work
  acc_first  : highest acc; ties -> lower work
The asymmetry under test: in the original, `cost` was applied to the adaptive
family with ref_acc = the event controller's own development accuracy, so the
comparator was *allowed* to be 0.5 pp worse. The symmetric arm applies the same
rule to the event family with ref_acc = the adaptive accuracy-first arm.
"""
from __future__ import annotations

import numpy as np

from isrm_sandbox.environments import frozen_bank as FB

THETAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)
GAMMAS = (0.8, 0.95, 0.99)
ADAPTIVE_CELLS = (
    (0.8, 0.0, 1),
    (0.8, 0.0, 16),
    (0.8, 0.03, 64),
    (0.8, 0.03, 256),
    (0.95, 0.03, 256),
    (0.8, 0.1, 4096),
    (0.95, 0.1, 512),
    (0.95, 0.1, 4096),
)
FINE_MARGINS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.1)
FINE_GAPS = (16, 32, 64, 128, 256, 512, 1024, 4096)
FINE_BETAS = (0.8, 0.95)
PERIODS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def event_family():
    return [(f"event[theta={th},gamma={g}]", lambda th=th, g=g: FB.CUSUMEventPolicy(theta=th, gamma=g)) for th in THETAS for g in GAMMAS]


def adaptive_family():
    fam = [
        (f"ewma[g={g},b={b},m={m},G={G}]", lambda g=g, b=b, m=m, G=G: FB.EWMAScanPolicy(beta=b, margin=m, max_gap=G, gamma=g))
        for g in GAMMAS
        for (b, m, G) in ADAPTIVE_CELLS
    ]
    fam.append(("ewma_strong[g=0.8,b=0.95,m=0.03,G=64]", lambda: FB.EWMAScanPolicy(beta=0.95, margin=0.03, max_gap=64, gamma=0.8)))
    fam.append(("ewma_quiet[g=0.8,b=0.95,m=1,G=4096]", lambda: FB.EWMAScanPolicy(beta=0.95, margin=1.0, max_gap=4096, gamma=0.8)))
    return fam


def fine_family():
    return [
        (f"ewma_fine[b={b},m={m},G={G}]", lambda b=b, m=m, G=G: FB.EWMAScanPolicy(beta=b, margin=m, max_gap=G, gamma=0.8))
        for b in FINE_BETAS
        for m in FINE_MARGINS
        for G in FINE_GAPS
    ]


def periodic_family():
    return [(f"periodic[P={P}]", lambda P=P: FB.PeriodicScanPolicy(period=P)) for P in PERIODS]


def dev_table(family, worlds, cfg):
    """Mean accuracy and work of every candidate over the development worlds."""
    rows = []
    for idx, (name, make) in enumerate(family):
        accs, works = [], []
        for w in worlds:
            r = FB.run_episode(w, cfg, make())
            accs.append(r["accuracy"])
            works.append(r["work"])
        rows.append({"id": idx, "name": name, "make": make, "acc": float(np.mean(accs)), "work": float(np.mean(works))})
    return rows


def select_cost(rows, ref_acc):
    best_acc = max(r["acc"] for r in rows)
    floor = max(ref_acc - 0.005, best_acc - 0.01)
    feasible = [r for r in rows if r["acc"] >= floor]
    if not feasible:
        return max(rows, key=lambda r: (r["acc"], -r["work"]))
    return min(feasible, key=lambda r: (r["work"], -r["acc"], r["id"]))


def select_budget(rows, ref_work):
    feasible = [r for r in rows if r["work"] <= ref_work + 1e-12]
    if not feasible:
        return min(rows, key=lambda r: (r["work"], -r["acc"]))
    return max(feasible, key=lambda r: (r["acc"], -r["work"]))


def select_acc_first(rows):
    return max(rows, key=lambda r: (r["acc"], -r["work"]))


def build(spec, seed_for):
    env = spec.environment
    cfg = FB.FrozenBankConfig(**env["bank"])
    n_dev = int(env["n_dev_worlds"])
    dev_worlds = [FB.make_world(np.random.default_rng(seed_for("dev", j)), cfg) for j in range(n_dev)]

    ev = dev_table(event_family(), dev_worlds, cfg)
    ad = dev_table(adaptive_family(), dev_worlds, cfg)
    fine = dev_table(fine_family(), dev_worlds, cfg)
    per = dev_table(periodic_family(), dev_worlds, cfg)

    fixed = next(r for r in ev if r["name"] == "event[theta=0.5,gamma=0.8]")
    ad_cost = select_cost(ad, ref_acc=fixed["acc"])
    ad_budget = select_budget(ad, ref_work=fixed["work"])
    ad_acc = select_acc_first(ad)
    fine_budget = select_budget(fine, ref_work=fixed["work"])
    ev_cost_sym = select_cost(ev, ref_acc=ad_acc["acc"])
    ev_budget_sym = select_budget(ev, ref_work=ad_acc["work"])
    per_budget = select_budget(per, ref_work=fixed["work"])

    arms = {
        "event_fixed": fixed["make"],
        "adaptive_cost_selected": ad_cost["make"],
        "adaptive_budget_selected": ad_budget["make"],
        "adaptive_accuracy_first": ad_acc["make"],
        "adaptive_budget_fine": fine_budget["make"],
        "event_cost_selected_sym": ev_cost_sym["make"],
        "event_budget_selected_sym": ev_budget_sym["make"],
        "periodic_budget_matched": per_budget["make"],
        "dense_scan": lambda: FB.DenseScanPolicy(),
        "never_scan": lambda: FB.NeverScanPolicy(),
    }
    selection = {
        "event_fixed": fixed["name"],
        "adaptive_cost_selected": ad_cost["name"],
        "adaptive_budget_selected": ad_budget["name"],
        "adaptive_accuracy_first": ad_acc["name"],
        "adaptive_budget_fine": fine_budget["name"],
        "event_cost_selected_sym": ev_cost_sym["name"],
        "event_budget_selected_sym": ev_budget_sym["name"],
        "periodic_budget_matched": per_budget["name"],
        "dev_acc_event_fixed": round(fixed["acc"], 5),
        "dev_work_event_fixed": round(fixed["work"], 4),
        "dev_acc_adaptive_accuracy_first": round(ad_acc["acc"], 5),
    }
    return cfg, arms, selection


def evaluate(spec, replicates, seed_for, metric: str):
    cfg, arms, selection = build(spec, seed_for)
    results = {name: [] for name in arms}
    results["oracle"] = []
    extras = {"work": {name: [] for name in list(arms) + ["oracle"]}, "final_quarter_accuracy_pct": {name: [] for name in list(arms) + ["oracle"]}, "scans": {name: [] for name in list(arms) + ["oracle"]}}
    for i in range(replicates):
        world = FB.make_world(np.random.default_rng(seed_for("env", i)), cfg)
        runs = {name: FB.run_episode(world, cfg, make()) for name, make in arms.items()}
        runs["oracle"] = FB.run_episode(world, cfg, FB.OracleScanPolicy(world["schedule"]))
        for name, r in runs.items():
            value = r[metric]
            results[name].append(100.0 * value if metric.endswith("accuracy") else value)
            extras["work"][name].append(r["work"])
            extras["final_quarter_accuracy_pct"][name].append(100.0 * r["final_quarter_accuracy"])
            extras["scans"][name].append(r["scans"])
    wanted = {a.name for a in spec.arms}
    results = {k: v for k, v in results.items() if k in wanted}
    extras["selection"] = selection
    return results, extras
