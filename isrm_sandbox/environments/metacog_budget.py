"""Shared-budget metacognition world for T03: six concurrent services, one replenished
credit budget, value-weighted deadlines, a public dependency DAG, and the shared
expected-value forecaster that every planner in the test must use.

Independent re-implementation of the written contract of
D:/Multi ISRM/isrm_metacognition_benchmark (MATHEMATICAL_CONTRACT.md, PREREGISTRATION.md,
EXECUTION_PROMPT.txt) as summarised in reports/REIMPLEMENTATION_SPECS.md. Nothing here
is copied from the original source tree; the import guard forbids importing it.

World (one *block* = one base seed; a block is instantiated in several *cells*)
---------------------------------------------------------------------------------
* n = 6 tasks, T = 256 ticks. Private latent z_i(t+1) = a z_i(t) + eps_i(t+1) + shock_i(t+1),
  a = 0.97, eps ~ N(0, sigma_eps_i^2) with sigma_eps in {0.02, 0.05} (three tasks each);
  z_i(0) ~ N(0, sigma_eps_i^2 / (1 - a^2)). Changing cells add shocks
  1[u < 0.01] N(0, 0.30^2); stable cells have none. The same noise tape serves every cell
  and every controller of a block (paired design).
* OBSERVE returns z_i(t) + sigma_obs_i n_i(t), sigma_obs in {0.05, 0.15} (three tasks each).
* Public DAG in a random topological order: the first task is a root; every later task
  draws 0, 1 or 2 parents (probabilities 0.25, 0.40, 0.35, capped by its position) among
  earlier tasks, so at most two parents per node and usually two or three roots. Signed
  coefficients a_ij with sum_j |a_ij| = 1. Coupling kappa = 0.35 (coupled); the
  independent condition has kappa = 0 and no edges at all. Targets: linear y_i = z_i + kappa sum_j a_ij y_j, nonlinear
  y_i = tanh(same), evaluated in topological order.
* Periods {8, 12, 16}, values {1, 3, 9}, computation lengths {1, 2, 3}, two tasks each,
  independently permuted; phase phi_i in [0, P_i); deadlines at ticks P_i + phi_i + k P_i.
* Tolerance tol_i = max(0.02, 0.5 sd_stable_i), sd_stable the analytic stationary sd of
  the true target under stable dynamics (times a development-fitted ratio r_nl for the
  nonlinear family). Correct iff |yhat_i - y_i(t_d)| <= tol_i; a missing answer is wrong.
* J = sum_due V_i correct_i / sum_due V_i.

Operations (at most one per task per tick; HOLD is the empty allocation)
-----------------------------------------------------------------------
OBSERVE(i) cost 1: scalar Kalman update (a, q_i = sigma_eps_i^2 + 0.01 x 0.09, r_i =
sigma_obs_i^2, prior N(0, q_i / (1 - a^2))); available to computations from the next tick.
COMPUTE(i) cost 1 per unit: legal iff no pending computation on i and every parent has a
published artifact; snapshots (m_i, v_i, t) of the tick-start filter and the parents'
published values; publishes yhat_i = f_i(m_snapshot, parent values) at the end of the tick
in which the L_i-th unit is paid. CONTINUE(i) cost 1 advances a pending computation.
CANCEL(i) cost 0.5 discards it. Budget B_0 = 6, B_max = 12, refill 1.5 (scarce) or 6
(ample) per tick, overflow discarded. Tick order: refill and predict, decide, execute from
the tick-start snapshot, publish, score deadlines due at this tick end.

State scores and forecaster (all planners share them)
-----------------------------------------------------
E = B / B_max. S = mean_i log(1 + v_i / v_ref_i), v_i the belief variance of y_i under the
artifact's consumed evidence {ancestor j -> (m_j, v_j, t_j)} propagated to now
(var_j(t) = a^{2k} v_j + q_j (1 - a^{2k}) / (1 - a^2)); linear: v_i = sum_j W_ij^2 var_j,
W = (I - kappa A)^-1; nonlinear: the Jacobian of the tanh map at the propagated means;
unpublished tasks sit at the prior v_ref_i. C = mean_i c_i with
c_i = min(1, |f_i(m_used, parents_used) - f_i(m_used, parents_latest)| / tol_i) for
published non-root tasks, else 0. The forecaster is a deterministic expected-value rollout
over H = 4 ticks: variances exact, means decay by a, a planned OBSERVE reduces variance
with the mean unchanged, planned computations publish f(propagated mean, predicted parent
values), budgets follow the refill rule. P(correct at deadline d) = Phi((tol - b) / sd) -
Phi((-tol - b) / sd) with b = yhat - mu_i(d) under the full-cache belief, then the
development-fitted recalibration sigmoid(c0 + c1 logit(P)). V_H sums V_i P over deadlines
in t..t+3; V_cont adds, per task, P at the next deadline after t+3 with no further action.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

NONE, OBS, COMP, CONT, CANC = 0, 1, 2, 3, 4
OP_COST = np.array([0.0, 1.0, 1.0, 1.0, 0.5])
OP_NAMES = ("HOLD", "OBSERVE", "COMPUTE", "CONTINUE", "CANCEL")
_ABSENT = -(10**8)  # evidence-map sentinel for "ancestor not consumed"


@dataclass
class MetacogConfig:
    n_tasks: int = 6
    ticks: int = 256
    a: float = 0.97
    p_shock: float = 0.01
    sigma_shock: float = 0.30
    sigma_eps: tuple = (0.02, 0.05)
    sigma_obs: tuple = (0.05, 0.15)
    kappa_coupled: float = 0.35
    B0: float = 6.0
    Bmax: float = 12.0
    refill_scarce: float = 1.5
    refill_ample: float = 6.0
    cancel_cost: float = 0.5
    tol_floor: float = 0.02
    tol_factor: float = 0.5
    periods: tuple = (8, 12, 16)
    values: tuple = (1, 3, 9)
    lengths: tuple = (1, 2, 3)
    horizon: int = 4
    max_parents: int = 2
    parent_probs: tuple = (0.25, 0.40, 0.35)  # P(0, 1, 2 parents) for every task after the first


CELLS = {
    f"{res}-{cpl}-{dyn}-{fam}": (res, cpl, dyn, fam)
    for res in ("scarce", "ample")
    for cpl in ("coupled", "independent")
    for dyn in ("stable", "changing")
    for fam in ("linear", "nonlinear")
}
PRIMARY_CELLS = (
    "scarce-coupled-stable-linear",
    "scarce-coupled-changing-linear",
    "scarce-coupled-stable-nonlinear",
    "scarce-coupled-changing-nonlinear",
)


# --------------------------------------------------------------------------- block
def make_block(rng: np.random.Generator, cfg: MetacogConfig = MetacogConfig()) -> dict:
    """Draw one base-seed block: structure, schedule and the exogenous noise tape.

    Everything a cell needs is drawn here so that every cell and every controller of the
    block sees identical structure and noise (the original's keyed tape).
    """
    n, T = cfg.n_tasks, cfg.ticks
    order = rng.permutation(n)
    parents: list[list[int]] = [[] for _ in range(n)]
    A = np.zeros((n, n))
    for k in range(1, n):
        i = int(order[k])
        n_par = min(k, int(rng.choice(np.arange(cfg.max_parents + 1), p=cfg.parent_probs)))
        if n_par == 0:
            continue
        chosen = rng.choice(order[:k], size=n_par, replace=False)
        mags = rng.dirichlet(np.ones(n_par)) if n_par > 1 else np.ones(1)
        signs = rng.choice([-1.0, 1.0], size=n_par)
        for j, mag, s in zip(chosen, mags, signs):
            parents[i].append(int(j))
            A[i, int(j)] = s * mag

    def balanced(vals):
        arr = np.repeat(np.asarray(vals, dtype=float), n // len(vals))
        return arr[rng.permutation(n)]

    sigma_eps = balanced(cfg.sigma_eps)
    sigma_obs = balanced(cfg.sigma_obs)
    periods = balanced(cfg.periods).astype(int)
    values = balanced(cfg.values)
    lengths = balanced(cfg.lengths).astype(int)
    phases = np.array([int(rng.integers(0, p)) for p in periods])
    return {
        "order": order,
        "parents": parents,
        "A": A,
        "sigma_eps": sigma_eps,
        "sigma_obs": sigma_obs,
        "periods": periods,
        "values": values,
        "lengths": lengths,
        "phases": phases,
        "z0": rng.normal(size=n),
        "eps": rng.normal(size=(T, n)),
        "shock_u": rng.random((T, n)),
        "shock_g": rng.normal(size=(T, n)),
        "meas": rng.normal(size=(T, n)),
    }


# --------------------------------------------------------------------------- episode
class Episode:
    """One block instantiated in one cell, with the public state a controller may read."""

    def __init__(self, block: dict, cell, cfg: MetacogConfig = MetacogConfig(), r_nl: float = 1.0):
        resources, coupling, dynamics, family = CELLS[cell] if isinstance(cell, str) else cell
        self.cfg = cfg
        n, T, a = cfg.n_tasks, cfg.ticks, cfg.a
        self.n, self.T, self.a = n, T, a
        self.cell = cell
        self.kappa = cfg.kappa_coupled if coupling == "coupled" else 0.0
        self.refill = cfg.refill_ample if resources == "ample" else cfg.refill_scarce
        self.nonlinear = family == "nonlinear"
        self.changing = dynamics == "changing"
        self.order = np.asarray(block["order"])
        if coupling == "coupled":
            self.parents = block["parents"]
            self.A = np.asarray(block["A"])
        else:  # the independent condition has no dependency edges (contract section 5: C == 0 exactly)
            self.parents = [[] for _ in range(n)]
            self.A = np.zeros((n, n))
        self.Amask = self.A != 0
        self.nonroot = self.Amask.any(1)
        par_idx = np.full((n, 2), n, dtype=int)  # index n = "no parent" row in extended arrays
        for i, ps in enumerate(self.parents):
            for k, j in enumerate(ps[:2]):
                par_idx[i, k] = j
        self.par_idx = par_idx
        self.W = np.linalg.inv(np.eye(n) - self.kappa * self.A)
        self.W2 = self.W**2
        self.L = np.asarray(block["lengths"], dtype=int)
        self.V = np.asarray(block["values"], dtype=float)
        sig_e, sig_o = np.asarray(block["sigma_eps"]), np.asarray(block["sigma_obs"])

        # private latent sources and truth
        sd0 = sig_e / np.sqrt(1.0 - a * a)
        if self.changing:
            shock = (block["shock_u"] < cfg.p_shock) * cfg.sigma_shock * block["shock_g"]
        else:
            shock = np.zeros((T, n))
        z = np.empty((T, n))
        z[0] = block["z0"] * sd0 + shock[0]
        for t in range(1, T):
            z[t] = a * z[t - 1] + sig_e * block["eps"][t] + shock[t]
        self.z = z
        self.y = self.truth(z)
        self.obs = z + sig_o * block["meas"]

        # deadlines
        self.deadlines = [np.arange(P + phi, T, P) for P, phi in zip(block["periods"], block["phases"])]
        due = np.zeros((T, n), dtype=bool)
        for i, ds in enumerate(self.deadlines):
            due[ds, i] = True
        self.due = due
        nxt = np.full((T + 1, n), T, dtype=int)  # first deadline strictly after tick t (T = none)
        for i, ds in enumerate(self.deadlines):
            for t in range(T):
                later = ds[ds > t]
                nxt[t, i] = int(later[0]) if later.size else T
        self.next_deadline = nxt

        # public filter parameters, normalisers and tolerances
        self.q = sig_e**2 + cfg.p_shock * cfg.sigma_shock**2
        self.r = sig_o**2
        self.v_prior = self.q / (1.0 - a * a)
        self.v_ref = self.W2 @ self.v_prior
        var_stable = sig_e**2 / (1.0 - a * a)
        scale = np.sqrt(self.W2 @ var_stable) * (r_nl if self.nonlinear else 1.0)
        self.tol = np.maximum(cfg.tol_floor, cfg.tol_factor * scale)

        # public state
        self.t = 0
        self.B = cfg.B0
        self.m = np.zeros(n)
        self.v = self.v_prior.copy()
        self.art_exists = np.zeros(n, dtype=bool)
        self.art_val = np.zeros(n)
        self.art_m_used = np.zeros(n)
        self.art_par_used = np.zeros((n, n))
        self.art_ev_m = np.zeros((n, n))
        self.art_ev_v = np.zeros((n, n))
        self.art_ev_t = np.full((n, n), _ABSENT, dtype=int)
        self.pend_active = np.zeros(n, dtype=bool)
        self.pend_left = np.zeros(n, dtype=int)
        self.pend_val = np.zeros(n)
        self.pend_m_used = np.zeros(n)
        self.pend_par_used = np.zeros((n, n))
        self.pend_ev_m = np.zeros((n, n))
        self.pend_ev_v = np.zeros((n, n))
        self.pend_ev_t = np.full((n, n), _ABSENT, dtype=int)

        # accounting
        self.value_due = 0.0
        self.value_correct = 0.0
        self.n_due = 0
        self.n_correct = 0
        self.n_missing = 0
        self.ops_paid = 0.0
        self.log: list[tuple] = []
        self._begin_tick()

    # ------------------------------------------------------------------ model helpers
    def truth(self, z: np.ndarray) -> np.ndarray:
        if not self.nonlinear:
            return z @ self.W.T
        y = np.zeros_like(z)
        for i in self.order:
            y[..., i] = np.tanh(z[..., i] + self.kappa * (y @ self.A[i]))
        return y

    def f_local(self, m: np.ndarray, parent_values: np.ndarray) -> np.ndarray:
        """Local one-level function f_i(m_i, parent values) for every task.

        `parent_values` is either (..., n) (the latest published values, read through the
        DAG) or (..., n, n) (the values each task consumed, one row per task).
        """
        if parent_values.ndim == m.ndim + 1:
            u = m + self.kappa * (self.A * parent_values).sum(-1)
        else:
            u = m + self.kappa * (parent_values @ self.A.T)
        return np.tanh(u) if self.nonlinear else u

    def belief_y(self, mu: np.ndarray, var: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mean and variance of the true targets under independent beliefs N(mu_j, var_j)."""
        if not self.nonlinear:
            return mu @ self.W.T, var @ self.W2.T
        # Jacobian dy/dz = (I - kappa D A)^-1 D by the topological recursion
        # J_i = D_i (e_i + kappa sum_{j in parents(i)} a_ij J_j); A is nilpotent so this is exact.
        n = self.n
        y = np.zeros_like(mu)
        J = np.zeros(mu.shape + (n,))
        eye = np.eye(n)
        for i in self.order:
            i = int(i)
            acc = eye[i]
            for j in self.parents[i]:
                acc = acc + self.kappa * self.A[i, j] * J[..., j, :]
            u = mu[..., i] + self.kappa * (y @ self.A[i])
            y_i = np.tanh(u)
            y[..., i] = y_i
            J[..., i, :] = (1.0 - y_i**2)[..., None] * acc
        return y, (J**2 * var[..., None, :]).sum(-1)

    def parents_published(self, art_exists: np.ndarray) -> np.ndarray:
        """(..., n) boolean: every parent of task i has a published artifact."""
        missing = (~art_exists)[..., None, :] & self.Amask
        return ~missing.any(-1)

    def merged_evidence(self, ev_m, ev_v, ev_t, m0, v0, tick):
        """Evidence maps every task would carry if computed now: latest snapshot per
        ancestor over the consumed parent chains, plus the task's own tick-start snapshot."""
        n = self.n
        lead = ev_t.shape[:-2]
        empty_t = np.full(lead + (1, n), _ABSENT, dtype=int)
        empty_f = np.zeros(lead + (1, n))
        ext_t = np.concatenate([ev_t, empty_t], axis=-2)
        ext_m = np.concatenate([ev_m, empty_f], axis=-2)
        ext_v = np.concatenate([ev_v, empty_f], axis=-2)
        p1, p2 = self.par_idx[:, 0], self.par_idx[:, 1]
        e1_t, e2_t = ext_t[..., p1, :], ext_t[..., p2, :]
        take1 = e1_t >= e2_t
        new_t = np.where(take1, e1_t, e2_t)
        new_m = np.where(take1, ext_m[..., p1, :], ext_m[..., p2, :])
        new_v = np.where(take1, ext_v[..., p1, :], ext_v[..., p2, :])
        idx = np.arange(n)
        new_t[..., idx, idx] = tick
        new_m[..., idx, idx] = m0
        new_v[..., idx, idx] = v0
        return new_m, new_v, new_t

    def S_from(self, art_exists, ev_m, ev_v, ev_t, tick) -> np.ndarray:
        """Uncertainty score from artifact evidence maps (..., n, n) at `tick`."""
        present = ev_t > _ABSENT // 2
        k = np.where(present, tick - ev_t, 0).astype(float)
        ak = self.a**k
        a2k = ak * ak
        var = np.where(present, a2k * ev_v + self.q * (1.0 - a2k) / (1.0 - self.a**2), 0.0)
        if not self.nonlinear:
            v_y = (self.W2 * var).sum(-1)
        else:
            mu = np.where(present, ak * ev_m, 0.0)
            _, var_full = self.belief_y(mu, var)
            idx = np.arange(self.n)
            v_y = var_full[..., idx, idx]
        v_y = np.where(art_exists, v_y, self.v_ref)
        return np.log1p(v_y / self.v_ref).mean(-1)

    def C_from(self, art_exists, art_m_used, art_par_used, art_val) -> np.ndarray:
        f_used = self.f_local(art_m_used, art_par_used)
        f_latest = self.f_local(art_m_used, art_val)
        c = np.minimum(1.0, np.abs(f_used - f_latest) / self.tol)
        c = np.where(art_exists & self.nonroot, c, 0.0)
        return c.mean(-1)

    def scores_now(self) -> tuple[float, float, float]:
        E = self.B / self.cfg.Bmax
        S = float(self.S_from(self.art_exists, self.art_ev_m, self.art_ev_v, self.art_ev_t, self.t))
        C = float(self.C_from(self.art_exists, self.art_m_used, self.art_par_used, self.art_val))
        return E, S, C

    # ------------------------------------------------------------------ dynamics
    def _begin_tick(self) -> None:
        self.B = min(self.cfg.Bmax, self.B + self.refill)
        self.m = self.a * self.m
        self.v = self.a * self.a * self.v + self.q

    def legal_compute(self) -> np.ndarray:
        return ~self.pend_active & self.parents_published(self.art_exists)

    def allocation_cost(self, alloc: dict) -> float:
        return float(sum(OP_COST[op] for op in alloc.values()))

    def execute(self, alloc: dict) -> None:
        """Execute one tick's allocation {task: op code}, publish, score, advance."""
        n, t = self.n, self.t
        if t >= self.T:
            raise RuntimeError("episode finished")
        cost = self.allocation_cost(alloc)
        if cost > self.B + 1e-9:
            raise ValueError(f"infeasible allocation at tick {t}: cost {cost} > budget {self.B}")
        m0, v0 = self.m.copy(), self.v.copy()
        legal_c = self.legal_compute()
        codes = np.zeros(n, dtype=np.int8)
        for i, op in alloc.items():
            codes[i] = op
            if op == OBS:
                K = self.v[i] / (self.v[i] + self.r[i])
                self.m[i] = self.m[i] + K * (self.obs[t, i] - self.m[i])
                self.v[i] = (1.0 - K) * self.v[i]
            elif op == COMP:
                if not legal_c[i]:
                    raise ValueError(f"illegal COMPUTE on task {i} at tick {t}")
                self.pend_active[i] = True
                self.pend_left[i] = self.L[i] - 1
                self.pend_val[i] = float(self.f_local(m0, self.art_val)[i])
                self.pend_m_used[i] = m0[i]
                self.pend_par_used[i] = self.art_val * self.Amask[i]
                ev_m, ev_v, ev_t = self.merged_evidence(self.art_ev_m, self.art_ev_v, self.art_ev_t, m0, v0, t)
                self.pend_ev_m[i], self.pend_ev_v[i], self.pend_ev_t[i] = ev_m[i], ev_v[i], ev_t[i]
            elif op == CONT:
                if not self.pend_active[i]:
                    raise ValueError(f"CONTINUE without pending work on task {i} at tick {t}")
                self.pend_left[i] -= 1
            elif op == CANC:
                if not self.pend_active[i]:
                    raise ValueError(f"CANCEL without pending work on task {i} at tick {t}")
                self.pend_active[i] = False
            elif op != NONE:
                raise ValueError(f"unknown op {op}")
        self.B -= cost
        self.ops_paid += cost
        self.log.append(tuple(int(c) for c in codes))
        pub = self.pend_active & (self.pend_left <= 0)
        if pub.any():
            self.art_exists[pub] = True
            self.art_val[pub] = self.pend_val[pub]
            self.art_m_used[pub] = self.pend_m_used[pub]
            self.art_par_used[pub] = self.pend_par_used[pub]
            self.art_ev_m[pub] = self.pend_ev_m[pub]
            self.art_ev_v[pub] = self.pend_ev_v[pub]
            self.art_ev_t[pub] = self.pend_ev_t[pub]
            self.pend_active[pub] = False
        for i in np.flatnonzero(self.due[t]):
            val = self.V[i]
            self.value_due += val
            self.n_due += 1
            if self.art_exists[i]:
                correct = abs(self.art_val[i] - self.y[t, i]) <= self.tol[i]
            else:
                correct = False
                self.n_missing += 1
            if correct:
                self.value_correct += val
                self.n_correct += 1
        self.t += 1
        if self.t < self.T:
            self._begin_tick()

    def result(self) -> dict:
        return {
            "J": self.value_correct / self.value_due if self.value_due else 0.0,
            "unweighted": self.n_correct / self.n_due if self.n_due else 0.0,
            "missing_rate": self.n_missing / self.n_due if self.n_due else 0.0,
            "ops_paid": self.ops_paid,
            "log": self.log,
        }


# --------------------------------------------------------------------------- forecaster
def candidates(ep: Episode) -> np.ndarray:
    """Public candidate generator: plans as (P, H, n) op-code arrays.

    Mandatory group: HOLD; per task OBSERVE, COMPUTE (or CONTINUE) to completion, CANCEL,
    CANCEL-then-COMPUTE, OBSERVE-then-COMPUTE; parent-then-child chains that fit in H.
    Fill group: pairwise simultaneous first steps and two bulk plans. Plans that are
    illegal under the current public state are not generated; budget feasibility is
    checked by the rollout. Generation is reward-neutral.
    """
    n, H = ep.n, ep.cfg.horizon
    legal_c = ep.legal_compute()
    pending = ep.pend_active
    plans: list[np.ndarray] = []

    def new() -> np.ndarray:
        return np.zeros((H, n), dtype=np.int8)

    def add_compute(plan: np.ndarray, i: int, start: int) -> bool:
        L = int(ep.L[i])
        if start + L > H:
            return False
        plan[start, i] = COMP
        for s in range(start + 1, start + L):
            plan[s, i] = CONT
        return True

    plans.append(new())
    for i in range(n):
        p = new()
        p[0, i] = OBS
        plans.append(p)
        if legal_c[i]:
            p = new()
            if add_compute(p, i, 0):
                plans.append(p)
            p = new()
            p[0, i] = OBS
            if add_compute(p, i, 1):
                plans.append(p)
        if pending[i]:
            left = int(ep.pend_left[i])
            p = new()
            for s in range(min(left, H)):
                p[s, i] = CONT
            plans.append(p)
            p = new()
            p[0, i] = CANC
            plans.append(p)
            if all(ep.art_exists[j] for j in ep.parents[i]):
                p = new()
                p[0, i] = CANC
                if add_compute(p, i, 1):
                    plans.append(p)
    for i in range(n):
        if pending[i]:
            continue
        for j in ep.parents[i]:
            if legal_c[j] and all(ep.art_exists[k] or k == j for k in ep.parents[i]):
                p = new()
                if add_compute(p, j, 0) and add_compute(p, i, int(ep.L[j])):
                    plans.append(p)
    for i in range(n):
        for j in range(i + 1, n):
            p = new()
            p[0, i] = OBS
            p[0, j] = OBS
            plans.append(p)
            if legal_c[j]:
                p = new()
                p[0, i] = OBS
                add_compute(p, j, 0)
                plans.append(p)
            if legal_c[i]:
                p = new()
                p[0, j] = OBS
                add_compute(p, i, 0)
                plans.append(p)
            if legal_c[i] and legal_c[j]:
                p = new()
                add_compute(p, i, 0)
                add_compute(p, j, 0)
                plans.append(p)
    p = new()
    p[0, :] = OBS
    plans.append(p)
    if legal_c.sum() >= 2:
        p = new()
        for i in np.flatnonzero(legal_c):
            add_compute(p, int(i), 0)
        plans.append(p)
    arr = np.stack(plans)
    _, idx = np.unique(arr.reshape(len(plans), -1), axis=0, return_index=True)
    return arr[np.sort(idx)]


class Forecaster:
    """Deterministic expected-value rollout shared by every planner (contract section 8)."""

    def __init__(self, c0: float = 0.0, c1: float = 1.0):
        self.c0, self.c1 = float(c0), float(c1)

    def pcal(self, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        logit = np.log(p / (1.0 - p))
        return 1.0 / (1.0 + np.exp(-(self.c0 + self.c1 * logit)))

    def p_correct(self, ep: Episode, mu, var, art_exists, art_val):
        """Calibrated P(artifact correct) at the tick of the belief (mu, var); 0 if no artifact."""
        mean_y, var_y = ep.belief_y(mu, var)
        sd = np.sqrt(np.maximum(var_y, 1e-12))
        b = art_val - mean_y
        pan = ndtr((ep.tol - b) / sd) - ndtr((-ep.tol - b) / sd)
        return np.where(art_exists, self.pcal(pan), 0.0), pan

    def rollout(self, ep: Episode, ops: np.ndarray) -> dict:
        """Features of every plan: E_after, S_after, C_after, V_H, V_cont, feasibility, first-step cost."""
        P, H, n = ops.shape
        cfg, a = ep.cfg, ep.a
        rep = lambda x: np.repeat(x[None], P, axis=0)  # noqa: E731
        B = np.full(P, ep.B)
        m, v = rep(ep.m), rep(ep.v)
        art_exists, art_val, art_m_used = rep(ep.art_exists), rep(ep.art_val), rep(ep.art_m_used)
        art_par_used = rep(ep.art_par_used)
        art_ev_m, art_ev_v, art_ev_t = rep(ep.art_ev_m), rep(ep.art_ev_v), rep(ep.art_ev_t)
        pend_active, pend_left, pend_val, pend_m_used = rep(ep.pend_active), rep(ep.pend_left), rep(ep.pend_val), rep(ep.pend_m_used)
        pend_par_used = rep(ep.pend_par_used)
        pend_ev_m, pend_ev_v, pend_ev_t = rep(ep.pend_ev_m), rep(ep.pend_ev_v), rep(ep.pend_ev_t)
        feasible = np.ones(P, dtype=bool)
        V_H = np.zeros(P)
        t = ep.t
        last = t
        for h in range(H):
            tick = t + h
            if tick >= ep.T:
                break
            last = tick
            if h > 0:  # the environment has already refilled and predicted for tick t
                B = np.minimum(cfg.Bmax, B + ep.refill)
                m = a * m
                v = a * a * v + ep.q
            step = ops[:, h, :]
            cost = OP_COST[step].sum(1)
            feasible &= B - cost >= -1e-9
            m0, v0 = m.copy(), v.copy()
            obs_mask = step == OBS
            v = np.where(obs_mask, v * ep.r / (v + ep.r), v)
            comp = step == COMP
            if comp.any():
                illegal = comp & (pend_active | ~ep.parents_published(art_exists))
                feasible &= ~illegal.any(1)
                fval = ep.f_local(m0, art_val)
                pend_left = np.where(comp, ep.L - 1, pend_left)
                pend_active = pend_active | comp
                pend_val = np.where(comp, fval, pend_val)
                pend_m_used = np.where(comp, m0, pend_m_used)
                pv = art_val[:, None, :] * ep.Amask
                pend_par_used = np.where(comp[..., None], pv, pend_par_used)
                ev_m, ev_v, ev_t = ep.merged_evidence(art_ev_m, art_ev_v, art_ev_t, m0, v0, tick)
                pend_ev_m = np.where(comp[..., None], ev_m, pend_ev_m)
                pend_ev_v = np.where(comp[..., None], ev_v, pend_ev_v)
                pend_ev_t = np.where(comp[..., None], ev_t, pend_ev_t)
            cont = step == CONT
            if cont.any():
                feasible &= ~(cont & ~pend_active).any(1)
                pend_left = np.where(cont, pend_left - 1, pend_left)
            canc = step == CANC
            if canc.any():
                feasible &= ~(canc & ~pend_active).any(1)
                pend_active = pend_active & ~canc
            B = B - cost
            pub = pend_active & (pend_left <= 0)
            if pub.any():
                art_exists = art_exists | pub
                art_val = np.where(pub, pend_val, art_val)
                art_m_used = np.where(pub, pend_m_used, art_m_used)
                art_par_used = np.where(pub[..., None], pend_par_used, art_par_used)
                art_ev_m = np.where(pub[..., None], pend_ev_m, art_ev_m)
                art_ev_v = np.where(pub[..., None], pend_ev_v, art_ev_v)
                art_ev_t = np.where(pub[..., None], pend_ev_t, art_ev_t)
                pend_active = pend_active & ~pub
            due = ep.due[tick]
            if due.any():
                pc, _ = self.p_correct(ep, m, v, art_exists, art_val)
                V_H += (pc * ep.V * due).sum(1)
        E_after = B / cfg.Bmax
        S_after = ep.S_from(art_exists, art_ev_m, art_ev_v, art_ev_t, last)
        C_after = ep.C_from(art_exists, art_m_used, art_par_used, art_val)
        nd = ep.next_deadline[last]
        valid = nd < ep.T
        k = (nd - last).astype(float)
        ak = a**k
        mu_c = ak * m
        var_c = ak * ak * v + ep.q * (1.0 - ak * ak) / (1.0 - a * a)
        pc, _ = self.p_correct(ep, mu_c, var_c, art_exists, art_val)
        V_cont = (pc * ep.V * valid).sum(1)
        return {
            "E_after": E_after,
            "S_after": S_after,
            "C_after": C_after,
            "V_H": V_H,
            "V_cont": V_cont,
            "feasible": feasible,
            "first_cost": OP_COST[ops[:, 0, :]].sum(1),
        }


def hold_forecast_pairs(ep: Episode, fc: Forecaster) -> list[tuple[float, int]]:
    """(uncalibrated P(correct), realised correctness) for the current artifacts at every
    deadline inside the horizon under no further action. Used only to fit the
    recalibration on development blocks; the realised label reads the private truth, which
    a training harness may do."""
    pairs = []
    t = ep.t
    for h in range(ep.cfg.horizon):
        tick = t + h
        if tick >= ep.T:
            break
        due = ep.due[tick] & ep.art_exists
        if not due.any():
            continue
        k = float(tick - t)
        ak = ep.a**k
        mu = ak * ep.m
        var = ak * ak * ep.v + ep.q * (1.0 - ak * ak) / (1.0 - ep.a**2)
        _, pan = fc.p_correct(ep, mu, var, ep.art_exists, ep.art_val)
        for i in np.flatnonzero(due):
            correct = int(abs(ep.art_val[i] - ep.y[tick, i]) <= ep.tol[i])
            pairs.append((float(pan[i]), correct))
    return pairs


def fit_recalibration(pairs: list[tuple[float, int]], iters: int = 50) -> tuple[float, float]:
    """Logistic recalibration Pcal = sigmoid(c0 + c1 logit(P)) by Newton's method."""
    if len(pairs) < 20:
        return 0.0, 1.0
    p = np.clip(np.array([x for x, _ in pairs]), 1e-4, 1 - 1e-4)
    y = np.array([c for _, c in pairs], dtype=float)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    for _ in range(iters):
        eta = X @ w
        mu = 1.0 / (1.0 + np.exp(-eta))
        g = X.T @ (mu - y) + 1e-3 * (w - np.array([0.0, 1.0]))
        Hs = (X * (mu * (1 - mu))[:, None]).T @ X + 1e-3 * np.eye(2)
        step = np.linalg.solve(Hs, g)
        w = w - step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def nonlinear_scale_ratio(blocks: list[dict], cfg: MetacogConfig = MetacogConfig()) -> float:
    """Development-fitted ratio of nonlinear to linear target sd under stable coupled dynamics."""
    ratios = []
    for b in blocks:
        lin = Episode(b, "scarce-coupled-stable-linear", cfg)
        nl = Episode(b, "scarce-coupled-stable-nonlinear", cfg)
        ratios.append(np.std(nl.y, axis=0) / np.maximum(np.std(lin.y, axis=0), 1e-12))
    return float(np.mean(ratios))


def run_episode(block: dict, cell, policy, cfg: MetacogConfig = MetacogConfig(), r_nl: float = 1.0, fc: Forecaster | None = None) -> dict:
    """Drive one policy through one block-cell episode. The policy exposes
    reset(ep) and decide(ep, fc) -> {task: op code}."""
    ep = Episode(block, cell, cfg, r_nl)
    fc = fc or Forecaster()
    policy.reset(ep)
    while ep.t < ep.T:
        alloc = policy.decide(ep, fc)
        ep.execute(alloc)
    return ep.result()
