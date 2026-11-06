"""Controllers for the shared-budget metacognition world (T03), written from the
written contract (MATHEMATICAL_CONTRACT.md section 7 and EXECUTION_PROMPT.txt section 10
of the original benchmark, via reports/REIMPLEMENTATION_SPECS.md).

Planners share one candidate generator and one forecaster (environments.metacog_budget)
and differ only in the score they maximise over the same feasible plans:

  U0(a)       = alpha E_after - beta (S_after - S_now) - gamma (C_after - C_now),
                alpha + beta + gamma = 1                                 (the ISRM utility)
  twin(a)     = w_E E_after - w_S S_after - w_C C_after                 (= U0 + constant)
  U_PLUS(a)   = V_H + V_cont + alpha E_after - beta dS - gamma dC       (extended utility)
  MPC(a)      = V_H + V_cont + lambda_B E_after                          (conventional)

Tie rule (contract 7.6): scores within 1e-9 tie; ties go to the lower first-step worker
cost, then to the canonical action order HOLD < OBSERVE < COMPUTE < CONTINUE < CANCEL over
tasks in label order. The winner's first-step allocation is executed.

FIXED_FAIR is the fixed comparator the prompt asks for: "a deterministic cyclic scheduler
that alternates observation and computation while respecting pending work and
feasibility; a competent fixed policy". Ours continues pending work first, then serves
tasks in cyclic topological order, alternating OBSERVE and COMPUTE per task, allocating as
many unit operations as the current budget affords. HOLD_ALL never acts.
"""
from __future__ import annotations

import numpy as np

from ..environments import metacog_budget as MB


class Policy:
    name = "policy"

    def reset(self, ep: MB.Episode) -> None:
        return None

    def decide(self, ep: MB.Episode, fc: MB.Forecaster) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class HoldAll(Policy):
    name = "hold_all"

    def decide(self, ep, fc):
        return {}


class FixedFair(Policy):
    name = "fixed_fair"

    def reset(self, ep):
        self.phase = np.zeros(ep.n, dtype=int)  # 0: observe next, 1: compute next
        self.ptr = 0

    def decide(self, ep, fc=None):
        n = ep.n
        alloc: dict[int, int] = {}
        budget = ep.B
        for i in range(n):
            if ep.pend_active[i] and budget >= 1.0 - 1e-9:
                alloc[i] = MB.CONT
                budget -= 1.0
        served_last = None
        for k in range(n):
            if budget < 1.0 - 1e-9:
                break
            i = int(ep.order[(self.ptr + k) % n])
            if i in alloc:
                continue
            if self.phase[i] == 0:
                alloc[i] = MB.OBS
                budget -= 1.0
                self.phase[i] = 1
                served_last = k
            elif all(ep.art_exists[j] for j in ep.parents[i]):
                alloc[i] = MB.COMP
                budget -= 1.0
                self.phase[i] = 0
                served_last = k
        if served_last is not None:
            self.ptr = (self.ptr + served_last + 1) % n
        return alloc


class RandomFeasible(Policy):
    """Development-only behaviour policy (mixture of fair service and random feasible
    plans) used to collect recalibration pairs. Never an arm."""

    name = "random_feasible"

    def __init__(self, rng: np.random.Generator, p_random: float = 0.5):
        self.rng, self.p_random = rng, p_random
        self.fair = FixedFair()

    def reset(self, ep):
        self.fair.reset(ep)

    def decide(self, ep, fc):
        if self.rng.random() >= self.p_random:
            return self.fair.decide(ep, fc)
        ops = MB.candidates(ep)
        feats = fc.rollout(ep, ops)
        idx = np.flatnonzero(feats["feasible"])
        if idx.size == 0:
            return {}
        choice = int(self.rng.choice(idx))
        return allocation_from(ops[choice, 0])


def allocation_from(first_step: np.ndarray) -> dict:
    return {int(i): int(op) for i, op in enumerate(first_step) if op != MB.NONE}


class Planner(Policy):
    """Receding-horizon planner over the shared candidate set and forecaster."""

    def __init__(self, rule: str, **params):
        self.rule = rule
        self.params = dict(params)
        if rule in ("U0", "U_PLUS"):
            a, b, g = params["alpha"], params["beta"], params["gamma"]
            if min(a, b, g) < 0:
                raise ValueError("coefficients must be non-negative")
            if rule == "U0":
                s = a + b + g
                if s <= 0:
                    raise ValueError("U0 needs a positive coefficient")
                self.params = {"alpha": a / s, "beta": b / s, "gamma": g / s}
        self.name = rule.lower()
        self.calls = 0

    def score(self, feats: dict, S_now: float, C_now: float) -> np.ndarray:
        p = self.params
        if self.rule == "U0":
            return p["alpha"] * feats["E_after"] - p["beta"] * (feats["S_after"] - S_now) - p["gamma"] * (feats["C_after"] - C_now)
        if self.rule == "LINEAR_EQUIVALENT":
            return p["w_E"] * feats["E_after"] - p["w_S"] * feats["S_after"] - p["w_C"] * feats["C_after"]
        if self.rule == "U_PLUS":
            return (
                feats["V_H"] + feats["V_cont"] + p["alpha"] * feats["E_after"]
                - p["beta"] * (feats["S_after"] - S_now) - p["gamma"] * (feats["C_after"] - C_now)
            )
        if self.rule == "CONVENTIONAL_MPC":
            return feats["V_H"] + feats["V_cont"] + p["lambda_B"] * feats["E_after"]
        raise ValueError(f"unknown rule {self.rule}")

    def decide(self, ep, fc):
        ops = MB.candidates(ep)
        feats = fc.rollout(ep, ops)
        feasible = feats["feasible"]
        if not feasible.any():
            return {}
        if self.rule in ("U0", "U_PLUS"):
            _, S_now, C_now = ep.scores_now()
        else:
            S_now = C_now = 0.0  # never read: MPC and the twin do not request the state scores
        score = np.where(feasible, self.score(feats, S_now, C_now), -np.inf)
        self.calls += 1
        best = select(score, feats["first_cost"], ops[:, 0, :])
        return allocation_from(ops[best, 0])


def select(score: np.ndarray, first_cost: np.ndarray, first_steps: np.ndarray, tol: float = 1e-9) -> int:
    """Argmax with the public tie rule."""
    best = score.max()
    tied = np.flatnonzero(score >= best - tol)
    if tied.size == 1:
        return int(tied[0])
    keys = [(float(first_cost[p]), tuple(int(x) for x in first_steps[p]), int(p)) for p in tied]
    return min(keys)[2]


# --------------------------------------------------------------------------- families
def u0_simplex(step: int = 6) -> list[dict]:
    """Every non-negative (alpha, beta, gamma) with alpha + beta + gamma = 1 on a 1/step grid."""
    grid = []
    for a in range(step + 1):
        for b in range(step + 1 - a):
            g = step - a - b
            grid.append({"alpha": a / step, "beta": b / step, "gamma": g / step})
    return grid


# MPC is the (beta, gamma) = (0, 0) member of the U_PLUS family (contract section 7.5), so the
# MPC lambda_B grid is exactly the beta = gamma = 0 slice of the U_PLUS grid. Calibrated on the
# same paired development episodes, a U_PLUS winner with beta = gamma = 0 is then MPC's own
# winner (identical decisions), and any U_PLUS - MPC difference is attributable to the S and
# C terms rather than to a different lambda_B. The original's values (lambda_B 3.4145; U_PLUS
# alpha 2, beta 4) are grid members.
LAMBDA_GRID = (0.0, 1.0, 2.0, 3.4145, 8.0)
U_PLUS_GRID = [
    {"alpha": al, "beta": be, "gamma": ga}
    for al in LAMBDA_GRID
    for be in (0.0, 2.0, 4.0, 8.0)
    for ga in (0.0, 2.0)
]
MPC_GRID = [{"lambda_B": lb} for lb in LAMBDA_GRID]


def make_policy(rule: str, params: dict) -> Policy:
    if rule == "FIXED_FAIR":
        return FixedFair()
    if rule == "HOLD_ALL":
        return HoldAll()
    return Planner(rule, **params)
