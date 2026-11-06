"""Controllers for the ball-avoidance arena (T04), written from the documented rules of the
Total ISRM v3 benchmark (MODEL_SPEC.txt, SOURCE_MAPPING.txt, DESIGN_AUDIT.txt, the
protocol constants and the engine's described behaviour). Nothing is imported or copied.

All planners share one observer (paid samples of every ball's position and velocity),
one motor planner (25 candidate target velocities ranked by forecast energy) and, for the
adaptive controllers, one model bank (an inertial predictor and a learned per-ball
turn-diffusion predictor with paid recalibration). They differ only in how predictive
*model operations* are chosen every 0.5 s:

  isrm              U = 0.4 E_op - 0.35 dS - 0.25 dC   (frozen weights of the original)
  energy            U = E_op                           (the original adaptive energy-only)
  energy_dwell      U = E_op - m 1[model identity changes], m = 0.25 x 0.1 / 0.4 = 0.0625:
                    ISRM's constant coherence-identity penalty expressed in energy units,
                    with no measurement of uncertainty or forecast disruption (the
                    dwell-only comparator of the frozen spec; the 1.5 s dwell and the
                    0.0005 hysteresis are already part of every adaptive controller)
  fixed             the v2 energy-only controller: inertial predictor, no model bank,
                    uncertainty-inflated collision margin, no governance fees
  isrm no-switch    the ISRM governor forbidden to change model identity (ablation)

E_op is the marginal expected resource gain of an operation under a shared
evidence-weighted reference forecast, divided by the 9-unit impact cost; dS is the change
in reducible prequential mismatch 1 - exp(-max(0, loss - best loss) / 4); dC is the mean
0.9 s forecast displacement plus spread change (normalised by 100, capped at 1 per ball)
plus 0.1 when the model identity changes. Operations: Continue, Switch (0.025),
Recalibrate (0.08 + fresh paid sample, holds the agent 0.3 s, then commits a fit of the
last 8 s of innovations). A 90-tick dwell follows any activation. Fees: observation 0.012,
model work 0.003 per sample, planning 0.005, assessment 0.008.
"""
from __future__ import annotations

import math

import numpy as np

from ..environments.ball_arena import DT, PHYS, TICKS

REG = {
    "decision_ticks": 6,
    "sample_ticks": 6,
    "low_sample_ticks": 18,
    "low_energy": 20.0,
    "observation_cost": 0.012,
    "planning_cost": 0.005,
    "reserve": 2.0,
    "reserve_fraction": 0.1,
    "surprise_memory": 0.65,
}
MODEL = {
    "assessment_ticks": 30,
    "assessment_cost": 0.008,
    "model_work_cost": 0.003,
    "switch_cost": 0.025,
    "recalibration_cost": 0.08,
    "recalibration_ticks": 18,
    "dwell_ticks": 90,
    "history_seconds": 8.0,
    "min_fit_seconds": 0.6,
    "min_fit_samples": 3,
    "issue_ticks": 18,
    "horizons": (18, 54),
    "variance_floor": 4.0,
    "initial_q": 0.08,
    "max_q": 4.0,
    "max_omega": 2.0,
    "score_memory": 0.85,
    "score_temperature": 2.0,
    "weight_floor": 0.03,
    "hysteresis": 0.0005,
    "coherence_identity": 0.1,
}
ISRM_WEIGHTS = (0.4, 0.35, 0.25)
KINDS = ("inertial", "turn", "shadow")
N_POINTS = 9  # forecast samples at 0.1 s intervals over the 0.9 s horizon
SUBSTEP = 2  # ticks per propagation sub-step inside a 0.1 s forecast interval

# candidate commands: hold plus 12 directions at two speeds
_ANG = np.arange(12) * np.pi / 6
CAND_VX = np.concatenate([[0.0], np.cos(_ANG) * 55.0, np.cos(_ANG) * 115.0])
CAND_VY = np.concatenate([[0.0], np.sin(_ANG) * 55.0, np.sin(_ANG) * 115.0])
N_CAND = len(CAND_VX)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def propagate(px, py, vx, vy, limit, ticks: int, omega=None):
    """Advance balls `ticks` ticks under optional per-ball angular drift, with wall reflections,
    in sub-steps of SUBSTEP ticks (the sandbox's own forecast integrator)."""
    px, py, vx, vy = px.copy(), py.copy(), vx.copy(), vy.copy()
    remaining = ticks
    while remaining > 0:
        k = min(SUBSTEP, remaining)
        h = k * DT
        if omega is not None:
            c, s = np.cos(omega * h), np.sin(omega * h)
            vx, vy = vx * c - vy * s, vx * s + vy * c
        px = px + vx * h
        py = py + vy * h
        d = np.hypot(px, py)
        out = d > limit
        if np.any(out):
            nx, ny = px[out] / d[out], py[out] / d[out]
            vn = vx[out] * nx + vy[out] * ny
            vxo, vyo = vx[out] - 2 * np.maximum(vn, 0) * nx, vy[out] - 2 * np.maximum(vn, 0) * ny
            vx[out], vy[out] = vxo, vyo
            px[out], py[out] = nx * (limit[out] - 1e-6), ny * (limit[out] - 1e-6)
        remaining -= k
    return px, py, vx, vy


def agent_paths(x, y, vx, vy):
    """Closed-form acceleration-limited paths of all candidates: positions (N_CAND, N_POINTS, 2)
    at 0.1 s intervals and the horizon energy expenditure (N_CAND,)."""
    dx, dy = CAND_VX - vx, CAND_VY - vy
    D = np.hypot(dx, dy)
    tau = D / PHYS["accel"]
    ticks = np.arange(1, N_POINTS * REG["decision_ticks"] + 1) * DT  # (54,)
    t = ticks[None, :]
    frac = np.where(D[:, None] > 0, np.minimum(1.0, t / np.maximum(tau[:, None], 1e-12)), 1.0)
    vxt = vx + dx[:, None] * frac
    vyt = vy + dy[:, None] * frac
    # position: integrate the piecewise-linear velocity analytically
    t_acc = np.minimum(t, tau[:, None])
    px = x + vx * t + dx[:, None] * (t_acc**2 / (2 * np.maximum(tau[:, None], 1e-12)) + np.maximum(t - tau[:, None], 0.0))
    py = y + vy * t + dy[:, None] * (t_acc**2 / (2 * np.maximum(tau[:, None], 1e-12)) + np.maximum(t - tau[:, None], 0.0))
    limit = PHYS["radius"] - PHYS["agent_radius"]
    d = np.hypot(px, py)
    scale = np.where(d > limit, limit / np.maximum(d, 1e-9), 1.0)
    px, py = px * scale, py * scale
    speed2 = vxt**2 + vyt**2
    expenditure = ((PHYS["basal"] + PHYS["motion"] * speed2 / PHYS["max_speed"] ** 2) * DT).sum(1)
    sel = np.arange(REG["decision_ticks"] - 1, N_POINTS * REG["decision_ticks"], REG["decision_ticks"])
    return np.stack([px[:, sel], py[:, sel]], axis=-1), expenditure


class ArenaController:
    """Observer, motor planner and (for adaptive kinds) model governor."""

    def __init__(self, kind: str = "isrm", governance: str = "full", weights=ISRM_WEIGHTS, margin: float | None = None):
        if kind not in ("isrm", "energy", "energy_dwell", "fixed"):
            raise ValueError(kind)
        self.kind = kind
        self.governance = governance
        self.alpha, self.beta, self.gamma = weights
        self.adaptive = kind != "fixed"
        if margin is None:
            margin = self.gamma * MODEL["coherence_identity"] / self.alpha if kind == "energy_dwell" else 0.0
        self.margin = margin
        self.name = kind

    # ------------------------------------------------------------------ state
    def reset(self, cell) -> None:
        self.cell = cell
        self.beliefs = None  # dict of arrays for every ball
        self.last_sample = None
        self.next_sample = 0
        self.interval = REG["sample_ticks"]
        self.surprise = 0.0
        self.active = "inertial"
        self.params = None  # turn parameters: (omega, q) per ball, None = default prior
        self.shadow = None
        self.history: list[tuple] = []  # (ball, tick, elapsed, turn)
        self.issued: list[dict] = []
        self.last_issue = None
        self.evidence = {k: {"loss": 0.0, "count": 0} for k in KINDS}
        self.switched_at = -MODEL["dwell_ticks"]
        self.next_assessment = MODEL["assessment_ticks"]
        self.pending = None
        self.revision = 0
        self.diag = {"switches": 0, "recalibrations": 0, "assessments": 0, "samples": 0, "deferred_samples": 0, "planning_deferred": 0,
                     "conserve": 0, "turn_ticks": 0, "decisions": 0, "observation_cost": 0.0, "model_cost": 0.0, "planning_cost": 0.0,
                     "governance_cost": 0.0, "recalibration_cost": 0.0, "verified_forecasts": 0}

    def diagnostics(self) -> dict:
        d = dict(self.diag)
        d["turn_fraction"] = d["turn_ticks"] / max(1, d["decisions"])
        return d

    # ------------------------------------------------------------------ model bank
    def _params(self, kind: str, N: int):
        if kind == "inertial":
            return np.zeros(N), np.zeros(N)
        src = self.shadow if kind == "shadow" else self.params
        if src is None:
            return np.zeros(N), np.full(N, MODEL["initial_q"])
        return src

    def _fit(self, tick: int, N: int):
        """Per-ball drift and diffusion from the last 8 s of reflection-corrected innovations."""
        omega0, q0 = self._params("turn", N)
        omega, q = omega0.copy(), q0.copy()
        support = np.zeros(N)
        if not self.history:
            return (omega, q), support
        hist = np.array(self.history, dtype=float)  # columns: ball, tick, elapsed, turn
        hist = hist[(tick - hist[:, 1]) * DT <= MODEL["history_seconds"]]
        self.history = [(int(b), int(t), e, u) for b, t, e, u in hist]
        if hist.size == 0:
            return (omega, q), support
        ball = hist[:, 0].astype(int)
        count = np.bincount(ball, minlength=N)
        duration = np.bincount(ball, weights=hist[:, 2], minlength=N)
        support = duration
        ok = (count >= MODEL["min_fit_samples"]) & (duration >= MODEL["min_fit_seconds"])
        if ok.any():
            turn_sum = np.bincount(ball, weights=hist[:, 3], minlength=N)
            om = np.clip(turn_sum / np.where(ok, duration, 1.0), -MODEL["max_omega"], MODEL["max_omega"])
            resid = (hist[:, 3] - om[ball] * hist[:, 2]) ** 2
            qq = np.clip(np.bincount(ball, weights=resid, minlength=N) / np.where(ok, duration, 1.0), 0.0, MODEL["max_q"])
            omega = np.where(ok, om, omega)
            q = np.where(ok, qq, q)
        return (omega, q), support

    def _excess(self, kind: str) -> float:
        e = self.evidence[kind]
        if e["count"] < 2:
            return 1.0
        valid = [v["loss"] for v in self.evidence.values() if v["count"] >= 2]
        return 1.0 - math.exp(-max(0.0, e["loss"] - min(valid)) / 4.0)

    def _weights(self) -> np.ndarray:
        scored = [self.evidence[k]["loss"] for k in KINDS if self.evidence[k]["count"] >= 2]
        minimum = min(scored + [0.0])
        losses = np.array([self.evidence[k]["loss"] if self.evidence[k]["count"] >= 2 else minimum + 2.0 for k in KINDS])
        w = np.maximum(MODEL["weight_floor"], np.exp(-(losses - losses.min()) / MODEL["score_temperature"]))
        return w / w.sum()

    def _variance(self, speed2, seconds, q):
        return MODEL["variance_floor"] + speed2 * q * seconds**3 / 3.0

    def _forecast(self, kind: str, age: int):
        """Ball forecasts under `kind` from the beliefs propagated to now: points (N_POINTS, N, 2), variance (N_POINTS, N)."""
        b = self.beliefs
        N = b["x"].size
        omega, q = self._params(kind, N)
        px, py, vx, vy = propagate(b["x"], b["y"], b["vx"], b["vy"], b["limit"], age, omega if kind != "inertial" else None)
        pts = np.empty((N_POINTS, N, 2))
        var = np.empty((N_POINTS, N))
        speed2 = b["vx"] ** 2 + b["vy"] ** 2
        for k in range(N_POINTS):
            px, py, vx, vy = propagate(px, py, vx, vy, b["limit"], REG["decision_ticks"], omega if kind != "inertial" else None)
            pts[k, :, 0], pts[k, :, 1] = px, py
            var[k] = self._variance(speed2, (age + (k + 1) * REG["decision_ticks"]) * DT, q)
        return pts, var

    # ------------------------------------------------------------------ observer
    def _observe(self, tick: int, hazards: np.ndarray, radii: np.ndarray, agent: dict, force: bool = False) -> bool:
        if not force and tick < self.next_sample:
            return False
        model_cost = MODEL["model_work_cost"] if self.adaptive else 0.0
        required = REG["observation_cost"] + model_cost + REG["planning_cost"] + PHYS["basal"] * REG["decision_ticks"] * DT
        if agent["energy"] < required:
            self.diag["deferred_samples"] += 1
            return False
        agent["energy"] -= REG["observation_cost"] + model_cost
        self.diag["observation_cost"] += REG["observation_cost"]
        self.diag["model_cost"] += model_cost
        N = hazards.shape[0]
        x, y, vx, vy = hazards[:, 0], hazards[:, 1], hazards[:, 2], hazards[:, 3]
        limit = PHYS["radius"] - radii
        if self.adaptive:
            self._score_issued(tick, x, y)
        if self.beliefs is None:
            uncertainty = np.zeros(N)
        else:
            b = self.beliefs
            elapsed_ticks = tick - self.last_sample
            elapsed = elapsed_ticks * DT
            ex, ey, evx, evy = propagate(b["x"], b["y"], b["vx"], b["vy"], limit, elapsed_ticks)
            ref_speed = np.full(N, self.cell.speed) if self.kind == "fixed" else np.hypot(b["vx"], b["vy"])
            residual = np.clip(np.hypot(x - ex, y - ey) / np.maximum(1.0, ref_speed * elapsed), 0.0, 1.0) if elapsed > 0 else np.zeros(N)
            uncertainty = b["uncertainty"] * REG["surprise_memory"] + residual * (1.0 - REG["surprise_memory"])
            if self.adaptive and elapsed > 0:
                speed = np.hypot(b["vx"], b["vy"])
                guard = speed * elapsed + 3.0
                near_wall = (np.hypot(b["x"], b["y"]) > limit - guard) | (np.hypot(x, y) > limit - guard)
                turns = np.arctan2(vy, vx) - np.arctan2(evy, evx)
                for i in np.flatnonzero(~near_wall):
                    self.history.append((int(i), tick, elapsed, _wrap(float(turns[i]))))
        self.beliefs = {"x": x.copy(), "y": y.copy(), "vx": vx.copy(), "vy": vy.copy(), "r": radii, "limit": limit, "uncertainty": uncertainty}
        self.surprise = float(uncertainty.max()) if N else 0.0
        self.last_sample = tick
        self.diag["samples"] += 1
        self.interval = REG["low_sample_ticks"] if (agent["energy"] < REG["low_energy"] and self.surprise <= 0.2) else REG["sample_ticks"]
        self.next_sample = tick + self.interval
        if self.adaptive:
            self._issue(tick)
        return True

    def _issue(self, tick: int) -> None:
        if self.last_issue is not None and tick - self.last_issue < MODEL["issue_ticks"]:
            return
        self.last_issue = tick
        b = self.beliefs
        N = b["x"].size
        speed2 = b["vx"] ** 2 + b["vy"] ** 2
        for kind in KINDS:
            omega, q = self._params(kind, N)
            px, py, vx, vy = b["x"], b["y"], b["vx"], b["vy"]
            prev = 0
            for horizon in MODEL["horizons"]:
                px, py, vx, vy = propagate(px, py, vx, vy, b["limit"], horizon - prev, omega if kind != "inertial" else None)
                prev = horizon
                self.issued.append({"kind": kind, "target": tick + horizon, "revision": self.revision, "x": px.copy(), "y": py.copy(),
                                    "var": self._variance(speed2, horizon * DT, q)})

    def _score_issued(self, tick: int, x: np.ndarray, y: np.ndarray) -> None:
        keep = []
        for f in self.issued:
            if f["target"] != tick:
                if f["target"] > tick:
                    keep.append(f)
                continue
            sq = (x - f["x"]) ** 2 + (y - f["y"]) ** 2
            loss = float(np.mean(np.log(f["var"]) + sq / (2 * f["var"])))
            e = self.evidence[f["kind"]]
            m = MODEL["score_memory"] if e["count"] else 0.0
            e["loss"] = m * e["loss"] + (1 - m) * loss
            e["count"] += 1
            if f["kind"] == self.active and f["revision"] == self.revision:
                self.diag["verified_forecasts"] += x.size
        self.issued = keep

    # ------------------------------------------------------------------ motor planner
    def _overhead(self, agent: dict, tick: int) -> float:
        horizon_ticks = int(round(PHYS["horizon"] / DT))
        first = max(REG["decision_ticks"], self.next_sample - tick)
        scans = 0 if first > horizon_ticks else 1 + (horizon_ticks - first) // self.interval
        per_scan = REG["observation_cost"] + (MODEL["model_work_cost"] if self.adaptive else 0.0)
        assess = PHYS["horizon"] / (MODEL["assessment_ticks"] * DT) * MODEL["assessment_cost"] if self.adaptive else 0.0
        return scans * per_scan + (horizon_ticks / REG["decision_ticks"]) * REG["planning_cost"] + assess

    def _evaluate(self, agent: dict, pts, var, tick: int):
        """Forecast total cost, E score and feasibility of every candidate under given ball forecasts."""
        paths, expenditure = agent_paths(agent["x"], agent["y"], agent["vx"], agent["vy"])  # (C, K, 2)
        gap = np.hypot(paths[:, :, None, 0] - pts[None, :, :, 0], paths[:, :, None, 1] - pts[None, :, :, 1]) - PHYS["agent_radius"] - self.beliefs["r"][None, None, :]
        if self.adaptive:
            margin = np.sqrt(81.0 + var)[None]  # (1, K, N)
        else:
            margin = (9.0 + 10.0 * self.beliefs["uncertainty"])[None, None, :]
        risk = np.where(gap <= 0, 1.0, np.exp(-(gap**2) / (2 * margin**2)))
        valid = ((np.arange(N_POINTS) + 1) * 0.1 > agent["cooldown"])[None, :, None]
        risk = np.where(valid, risk, 0.0).max(axis=1)  # (C, N)
        combined = 1.0 - np.prod(1.0 - risk, axis=1)
        total = expenditure + PHYS["hit_cost"] * combined + self._overhead(agent, tick)
        remaining = agent["energy"] - total
        reserve = min(REG["reserve"], max(0.0, agent["energy"]) * REG["reserve_fraction"])
        feasible = remaining >= reserve
        E = np.log1p(np.maximum(0.0, remaining)) / np.log(101.0)
        return {"total": total, "E": E, "feasible": feasible}

    @staticmethod
    def _best(ev: dict) -> int:
        if ev["feasible"].any():
            E = np.where(ev["feasible"], ev["E"], -np.inf)
            return int(np.argmax(E))
        return int(np.argmin(ev["total"]))

    # ------------------------------------------------------------------ governor
    def _complete_recalibration(self, tick: int, hazards, radii, agent) -> None:
        if self.pending is None or tick < self.pending["complete"]:
            return
        if not self._observe(tick, hazards, radii, agent, force=True):
            self.pending["blocked"] = True
            self.diag["conserve"] += 1
            return
        old = self.active
        N = hazards.shape[0]
        self.params, _ = self._fit(tick, N)
        self.shadow = (self.params[0].copy(), self.params[1].copy())
        self.active = "turn"
        self.revision += 1
        self.switched_at = tick
        self.pending = None
        self.next_assessment = tick + MODEL["dwell_ticks"]
        self.diag["recalibrations"] += 1
        if old != "turn":
            self.diag["switches"] += 1

    def _govern(self, tick: int, agent: dict) -> None:
        if self.pending is not None or tick < self.next_assessment:
            return
        self.next_assessment = tick + MODEL["assessment_ticks"]
        reserve = min(REG["reserve"], max(0.0, agent["energy"]) * REG["reserve_fraction"])
        floor = reserve + PHYS["basal"] * MODEL["recalibration_ticks"] * DT + REG["planning_cost"] + REG["observation_cost"] + MODEL["model_work_cost"]
        if agent["energy"] < MODEL["assessment_cost"] + floor:
            self.diag["conserve"] += 1
            return
        agent["energy"] -= MODEL["assessment_cost"]
        self.diag["governance_cost"] += MODEL["assessment_cost"]
        self.diag["assessments"] += 1
        N = self.beliefs["x"].size
        self.shadow, support = self._fit(tick, N)
        age = tick - self.last_sample
        forecasts = {k: self._forecast(k, age) for k in KINDS}
        evals = {k: self._evaluate(agent, forecasts[k][0], forecasts[k][1], tick) for k in KINDS}
        w = self._weights()
        ref_cost = sum(w[i] * evals[k]["total"] for i, k in enumerate(KINDS))  # (C,)
        current_action = self._best(evals[self.active])
        current_cost = ref_cost[current_action]
        current_S = self._excess(self.active)
        other = "turn" if self.active == "inertial" else "inertial"
        dwell = tick - self.switched_at >= MODEL["dwell_ticks"]
        no_switch = self.governance == "no-switch"
        ops = [
            {"op": "Continue", "target": self.active, "kind": self.active, "cost": 0.0, "allowed": True},
            {"op": "Switch", "target": other, "kind": other, "cost": MODEL["switch_cost"], "allowed": dwell and not no_switch},
            {"op": "Recalibrate", "target": "turn", "kind": "shadow",
             "cost": MODEL["recalibration_cost"] + (0.0 if self.active == "turn" else MODEL["switch_cost"]) + REG["observation_cost"] + MODEL["model_work_cost"],
             "allowed": dwell and (not no_switch or self.active == "turn") and bool(np.any(support >= MODEL["min_fit_seconds"]))},
        ]
        for op in ops:
            cand = self._best(evals[op["kind"]])
            delay = 0.0
            if op["op"] == "Recalibrate":
                delay = max(0.0, ref_cost[0] - ref_cost[cand]) * MODEL["recalibration_ticks"] * DT / PHYS["horizon"]
            e_val = (current_cost - ref_cost[cand] - op["cost"] - delay) / PHYS["hit_cost"]
            dS = self._excess(op["kind"]) - current_S
            identity = op["target"] != self.active
            if op["op"] == "Continue":
                dC = 0.0
            else:
                a_pts, a_var = forecasts[self.active]
                b_pts, b_var = forecasts[op["kind"]]
                disp = np.hypot(a_pts[-1, :, 0] - b_pts[-1, :, 0], a_pts[-1, :, 1] - b_pts[-1, :, 1])
                spread = np.abs(np.sqrt(a_var[-1]) - np.sqrt(b_var[-1]))
                dC = float(np.minimum(1.0, (disp + spread) / 100.0).mean()) + (MODEL["coherence_identity"] if identity else 0.0)
            if self.kind == "isrm" and (self.beta > 0 or self.gamma > 0):
                U = self.alpha * e_val - self.beta * dS - self.gamma * dC
            elif self.kind == "energy_dwell":
                U = e_val - (self.margin if identity else 0.0)
            else:  # energy-only, and ISRM with beta = gamma = 0 (exact zero-extra-weight equivalence)
                U = e_val
            op.update(U=U, feasible=op["allowed"] and agent["energy"] - op["cost"] >= floor)
        feasible = [o for o in ops if o["feasible"]]
        best = max(feasible, key=lambda o: o["U"]) if feasible else ops[0]
        selected = best if best["U"] > MODEL["hysteresis"] else ops[0]
        if selected["op"] == "Switch":
            agent["energy"] -= selected["cost"]
            self.diag["governance_cost"] += selected["cost"]
            self.active = selected["target"]
            self.switched_at = tick
            self.revision += 1
            self.diag["switches"] += 1
        elif selected["op"] == "Recalibrate":
            cost = selected["cost"] - REG["observation_cost"] - MODEL["model_work_cost"]
            agent["energy"] -= cost
            self.diag["recalibration_cost"] += MODEL["recalibration_cost"]
            if self.active != "turn":
                self.diag["governance_cost"] += MODEL["switch_cost"]
            self.pending = {"complete": tick + MODEL["recalibration_ticks"], "blocked": False}

    # ------------------------------------------------------------------ decision
    def decide(self, tick: int, hazards: np.ndarray, radii: np.ndarray, agent: dict):
        self.diag["decisions"] += 1
        if self.active == "turn":
            self.diag["turn_ticks"] += 1
        if self.adaptive:
            self._complete_recalibration(tick, hazards, radii, agent)
            if self.pending is None or not self.pending.get("blocked"):
                self._observe(tick, hazards, radii, agent)
            self._govern(tick, agent)
            if self.pending is not None:
                return 0.0, 0.0  # hold while the paid model update awaits its fresh sample
        else:
            self._observe(tick, hazards, radii, agent)
        if self.beliefs is None or agent["energy"] < REG["planning_cost"] + PHYS["basal"] * REG["decision_ticks"] * DT:
            self.diag["planning_deferred"] += 1
            return 0.0, 0.0
        agent["energy"] -= REG["planning_cost"]
        self.diag["planning_cost"] += REG["planning_cost"]
        age = tick - self.last_sample
        pts, var = self._forecast(self.active if self.adaptive else "inertial", age)
        ev = self._evaluate(agent, pts, var, tick)
        c = self._best(ev)
        return float(CAND_VX[c]), float(CAND_VY[c])


def make_controller(arm: str) -> ArenaController:
    if arm == "isrm_full":
        return ArenaController("isrm")
    if arm == "adaptive_energy_dwell":
        return ArenaController("energy_dwell")
    if arm == "adaptive_energy_only":
        return ArenaController("energy")
    if arm == "fixed_energy_only":
        return ArenaController("fixed")
    if arm == "isrm_no_switching":
        return ArenaController("isrm", governance="no-switch")
    raise ValueError(arm)
