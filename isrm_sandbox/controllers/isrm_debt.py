"""Decision-debt arbitration controller, re-implemented from ISRM_manuscript1.md
section 2.2 with the parameters of section 10.7, its matched twin (lambda_eps = 0),
and two conventional state-carrying alternatives the manuscript does not consider.

Stage 1 (escalation to access), as written:
    escalate iff  dL_t + lambda_eps * eps_t > c_broadcast(E_cap)
    dL_t = max(0, E[L | retain] - min_a E[L | a]) over horizon H with terminal
           actions {continue, revise}: E[L|retain] = u H q, E[L|revise] = u H (1-q) + c_revise
    c_broadcast(E) = c0 (1 + 0.5 (1 - E))   # 6.0 at full capacity, rising with depletion
        (the manuscript fixes the value at full capacity and the direction; the
         linear form with slope 0.5 is the one the original simulation used)

Stage 2 (arbitration after access): feasibility is a veto
    F = {ACT, HOLD} + {CHECK if E >= e_check} + {REVISE if E >= e_revise}
    J(ACT)    = u H q
    J(REVISE) = c_revise + u H (1 - q)
    J(CHECK)  = u q + c_check + E_o[V_term(q_{t+1|o})]      (non-circular EVOI)
    J(HOLD)   = u q + c_hold + V_term(q)
    V_term(q) = min(u H q, u H (1-q) + c_revise)
    a = argmin J over F, ties by least irreversibility (HOLD < CHECK < ACT < REVISE)

Debt dynamics, as written (identical in every arm; only lambda_eps differs):
    alpha_t = max(0, q_t - 1/2) u 1[not settled at t]
    sigma_t = min(s (1-rho) eps_t, (1-rho) eps_t) 1[REVISE at t and verified]
    eps_{t+1} = max(0, (1-rho) eps_t + alpha_t - sigma_t)
The equation runs on every trial, including the test episode (the original
simulation froze debt during the test; the manuscript's equations do not).

Belief: change-point filter q = P(adopted model wrong), predicted forward with the
filter hazard and updated by the emission likelihood. A context reset sets q = 1/2
and the adopted model; it does not touch eps, nor the alternatives' carried state.

Alternatives (lambda_eps = 0):
  * adaptive_hazard: learns the hazard from how often the adopted model is surprised
    (plain parameter adaptation); the estimate persists across the context reset.
  * leaky_accumulator: adds lambda_A * A_t to the escalation test, A_t a leaky sum of
    consequential discrepancy max(0, q - 1/2) that is never settled by revision.
    This is decision debt without the settlement clause.

The bounded self-model update (eta, B), the risk term lambda_S S(a) and the
self-forecast penalty lambda_z Phi are omitted: with capacity constant within an
episode they cannot change the escalation trial, and the original simulation also
never executed them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..environments.changepoint_task import Episode, model_prediction

ACT, CHECK, REVISE, HOLD = "ACT", "CHECK", "REVISE", "HOLD"
IRREVERSIBILITY = {HOLD: 0, CHECK: 1, ACT: 2, REVISE: 3}


@dataclass
class DebtParams:
    hazard: float = 0.05
    accuracy: float = 0.62
    horizon: int = 8
    u: float = 1.0
    c_broadcast: float = 6.0
    broadcast_slope: float = 0.5
    c_check: float = 0.35
    c_revise: float = 1.0
    c_hold: float = 0.25
    e_check: float = 0.25
    e_revise: float = 0.55
    rho: float = 0.10
    lambda_eps: float = 0.5
    settle_frac: float = 1.0
    freeze_debt_in_test: bool = False
    adaptive_hazard: bool = False
    hazard_kappa: float = 0.02
    hazard_min: float = 0.01
    hazard_max: float = 0.30
    leaky_accumulator: bool = False
    lambda_acc: float = 0.5
    acc_rho: float = 0.10


class DebtController:
    def __init__(self, p: DebtParams):
        self.p = p
        self.eps = 0.0
        self.acc = 0.0
        self.h_est = p.hazard
        self.q = 0.5
        self.model = 0
        self.E_cap = 1.0

    # ------------------------------------------------------------- belief
    def reset_context(self, model: int = 0) -> None:
        """Explicit context reset: belief re-initialised, adopted model as given."""
        self.q = 0.5
        self.model = model

    def _hazard(self) -> float:
        return self.h_est if self.p.adaptive_hazard else self.p.hazard

    def _likelihoods(self, o: int) -> tuple[float, float]:
        pred = model_prediction(self.model)
        acc = self.p.accuracy
        p_unchanged = acc if o == pred else 1.0 - acc
        return 1.0 - p_unchanged, p_unchanged  # (P(o | model wrong), P(o | model right))

    def update_belief(self, o: int) -> None:
        q = self.q + (1.0 - self.q) * self._hazard()
        pw, pr = self._likelihoods(o)
        num = q * pw
        self.q = num / (num + (1.0 - q) * pr)
        if self.p.adaptive_hazard:
            surprised = 1.0 if o != model_prediction(self.model) else 0.0
            excess = max(0.0, surprised - (1.0 - self.p.accuracy))
            target = min(self.p.hazard_max, self.p.hazard_min + excess)
            self.h_est = float(
                np.clip(self.h_est + self.p.hazard_kappa * (target - self.h_est), self.p.hazard_min, self.p.hazard_max)
            )

    # ------------------------------------------------------------- values
    def v_term(self, q: float) -> float:
        p = self.p
        return min(p.u * p.horizon * q, p.u * p.horizon * (1.0 - q) + p.c_revise)

    def delta_L(self) -> float:
        p = self.p
        retain = p.u * p.horizon * self.q
        best = min(retain, p.u * p.horizon * (1.0 - self.q) + p.c_revise)
        return max(0.0, retain - best)

    def c_broadcast(self) -> float:
        return self.p.c_broadcast * (1.0 + self.p.broadcast_slope * (1.0 - self.E_cap))

    def escalates(self) -> bool:
        carried = self.p.lambda_eps * self.eps
        if self.p.leaky_accumulator:
            carried += self.p.lambda_acc * self.acc
        return self.delta_L() + carried > self.c_broadcast()

    def expected_posterior_vterm(self) -> float:
        q_pred = self.q + (1.0 - self.q) * self._hazard()
        total = 0.0
        for o in (0, 1):
            pw, pr = self._likelihoods(o)
            p_o = q_pred * pw + (1.0 - q_pred) * pr
            q_post = q_pred * pw / p_o
            total += p_o * self.v_term(q_post)
        return total

    def feasible(self) -> list[str]:
        f = [ACT, HOLD]
        if self.E_cap >= self.p.e_check:
            f.append(CHECK)
        if self.E_cap >= self.p.e_revise:
            f.append(REVISE)
        return f

    def choose(self) -> str:
        p = self.p
        J = {
            ACT: p.u * p.horizon * self.q,
            REVISE: p.c_revise + p.u * p.horizon * (1.0 - self.q),
            CHECK: p.u * self.q + p.c_check + self.expected_posterior_vterm(),
            HOLD: p.u * self.q + p.c_hold + self.v_term(self.q),
        }
        return min(self.feasible(), key=lambda a: (round(J[a], 12), IRREVERSIBILITY[a]))

    # ------------------------------------------------------------- carried state
    def update_debt(self, settled: bool) -> None:
        p = self.p
        alpha = max(0.0, self.q - 0.5) * p.u * (0.0 if settled else 1.0)
        carried = (1.0 - p.rho) * self.eps
        sigma = min(p.settle_frac * carried, carried) if settled else 0.0
        self.eps = max(0.0, carried + alpha - sigma)
        if p.leaky_accumulator:
            self.acc = (1.0 - p.acc_rho) * self.acc + max(0.0, self.q - 0.5) * p.u


def run_episode(ctrl: DebtController, ep: Episode, capacity: float, is_test: bool = False) -> dict:
    """Run one episode: escalate, arbitrate, revise, settle. Returns the first escalation trial.

    Preparation episodes end when a verified revision resolves the discrepancy.
    The test episode ends at the first escalation (the endpoint of the design).
    """
    ctrl.E_cap = capacity
    ctrl.reset_context(model=1 - int(ep.rule[0]))  # adopted model starts wrong
    first_escalation = None
    loss = 0.0
    revisions = 0
    for t in range(len(ep.obs)):
        o = int(ep.obs[t])
        ctrl.update_belief(o)
        loss += ctrl.p.u if ctrl.model != ep.rule[t] else 0.0
        settled = False
        if ctrl.escalates():
            if first_escalation is None:
                first_escalation = t + 1
                if is_test:
                    break
            a = ctrl.choose()
            if a == REVISE:
                revisions += 1
                new_model = 1 - ctrl.model
                verified = new_model == ep.rule[t]
                ctrl.model = new_model
                ctrl.q = 0.5
                settled = verified
        if not (is_test and ctrl.p.freeze_debt_in_test):
            ctrl.update_debt(settled)
        if settled:
            break  # discrepancy resolved: the preparation episode ends
    if first_escalation is None:
        first_escalation = len(ep.obs) + 1  # censored: no escalation within the episode
    return {"loss": loss, "revisions": revisions, "first_escalation": first_escalation, "eps_end": ctrl.eps}


def history_effect(
    params: DebtParams,
    participant: dict,
    resolved_capacity: float = 1.0,
    deferred_capacity: float = 0.35,
    prep_episodes: int | None = None,
) -> dict:
    """Trials to escalation at test after resolved vs deferred preparation; same streams."""
    out: dict = {}
    for cond, cap in (("resolved", resolved_capacity), ("deferred", deferred_capacity)):
        ctrl = DebtController(params)
        preps = participant["prep"] if prep_episodes is None else participant["prep"][:prep_episodes]
        for ep in preps:
            run_episode(ctrl, ep, capacity=cap)
        carried = {"eps": ctrl.eps, "acc": ctrl.acc, "h_est": ctrl.h_est}
        test = run_episode(ctrl, participant["test"], capacity=1.0, is_test=True)
        censored = test["first_escalation"] > len(participant["test"].obs)
        out[cond] = {"escalation": test["first_escalation"], "censored": censored, **carried}
    out["effect"] = out["resolved"]["escalation"] - out["deferred"]["escalation"]
    return out
