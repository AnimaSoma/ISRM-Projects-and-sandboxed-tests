"""Reserve-and-supply toy plant for T08 (the "dual weights" study), re-implemented from the
written protocol and reports of D:/ISRM Dual Weights (V3_PROTOCOL.md, V3_REPORT.md,
README.md, REPORT.md) as summarised in reports/REIMPLEMENTATION_SPECS.md. No code of the
original is imported or copied.

Plant
-----
* Reserve R in [0, 1], R(0) = 0.8. Supply 0.10 per tick except during 40-tick deficits:
  the first starts at t = 200, the next about 400 +/- 40 ticks after the previous onset.
* True cost per unit of effort is 0.15 kappa_true, kappa_true starting at kappa_0 = 1.
  Regimes: R1 kappa never changes; R2 kappa jumps to 0.6 or 1.6 (random) 150..250 ticks
  after a deficit onset; R3 jumps at the deficit onset. The deficit schedule and the
  draws are shared by the three regimes of a replicate (paired worlds).
* Actions, in this order (strict first-minimum tie rule): Abstain (effort 0), Act 0.5,
  Act 1, Sample (fixed cost 0.01; stores the current supply as s_known and resets
  staleness), Update (fixed cost 0.10; oracle refit kappa_model := kappa_true and resets
  the estimated consistency meter).
* Reserve-funded feasibility (the v3 primary cell): Sample needs pre-tick R >= 0.01,
  Update needs R >= 0.10. Effort is curtailed to what reserve plus incoming supply can
  fund: effort_real = min(effort, (R + supply - fixed) / (0.15 kappa_true)).
  R' = clip(R + supply - fixed - 0.15 kappa_true effort_real, 0, 1). Work = effort_real.
* Meters (post-action): E residual = 0.30 - R'; staleness S' = 0 after Sample else
  min(1, S + 0.10), threshold 0.20; C_est' = 0 after Update else 0.7 C_est + 0.3 min(1,
  |actual - predicted spend| / predicted spend) (0 when nothing was predicted),
  threshold 0.10. C_true = min(1, |kappa_model - kappa_true| / kappa_0) is reported, never
  controlled. Violation depths are means of the positive threshold residuals.

Predicted residuals (one-step, under the controller's model)
------------------------------------------------------------
r_hat_E(a) = 0.30 - clip(R + s_known - 0.15 kappa_model effort(a) - fixed(a), 0, 1)
r_hat_S(a) = (0 if Sample else min(1, S + 0.10)) - 0.20
r_hat_C(a) = (0 if Update else 0.7 C_est) - 0.10
Signed residuals in the primary cell; the hinged variant max(0, r_hat) is kept as an
option but not used by the frozen T08 arms.

The initial supply estimate s_known is the nominal 0.10 (the controller is assumed to
know the design supply before any deficit); the original is silent on this value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ABSTAIN, ACT_HALF, ACT_FULL, SAMPLE, UPDATE = 0, 1, 2, 3, 4
ACTION_NAMES = ("Abstain", "Act0.5", "Act1", "Sample", "Update")
EFFORT = np.array([0.0, 0.5, 1.0, 0.0, 0.0])
FIXED = np.array([0.0, 0.0, 0.0, 0.01, 0.10])
N_ACTIONS = 5


@dataclass
class PlantConfig:
    ticks: int = 4000
    R0: float = 0.8
    supply: float = 0.10
    deficit_len: int = 40
    first_deficit: int = 200
    deficit_gap: int = 400
    deficit_jitter: int = 40
    unit_cost: float = 0.15
    kappa0: float = 1.0
    kappa_targets: tuple = (0.6, 1.6)
    r2_lag: tuple = (150, 250)
    e_min: float = 0.30
    s_rate: float = 0.10
    s_threshold: float = 0.20
    c_decay: float = 0.7
    c_threshold: float = 0.10
    sample_cost: float = 0.01
    update_cost: float = 0.10
    sample_floor: float = 0.01
    update_floor: float = 0.10
    s_known0: float = 0.10


THRESHOLDS = {"E": 0.30, "S": 0.20, "C": 0.10}


def make_world(rng: np.random.Generator, cfg: PlantConfig = PlantConfig()) -> dict:
    """Deficit schedule and kappa paths for R1, R2 and R3 (shared draws, paired)."""
    T = cfg.ticks
    supply = np.full(T, cfg.supply)
    onsets = []
    t = cfg.first_deficit
    while t < T:
        onsets.append(t)
        supply[t : min(T, t + cfg.deficit_len)] = 0.0
        t += cfg.deficit_gap + int(rng.integers(-cfg.deficit_jitter, cfg.deficit_jitter + 1))
    kappa = {"R1": np.full(T, cfg.kappa0)}
    k2 = np.full(T, cfg.kappa0)
    k3 = np.full(T, cfg.kappa0)
    for onset in onsets:
        target = float(rng.choice(cfg.kappa_targets))
        lag = int(rng.integers(cfg.r2_lag[0], cfg.r2_lag[1] + 1))
        if onset + lag < T:
            k2[onset + lag :] = target
        k3[onset:] = target
    kappa["R2"], kappa["R3"] = k2, k3
    return {"supply": supply, "onsets": np.array(onsets, dtype=int), "kappa": kappa, "T": T}


class Plant:
    """Scalar plant driven one tick at a time by a controller's action index."""

    def __init__(self, world: dict, regime: str, cfg: PlantConfig = PlantConfig()):
        self.cfg = cfg
        self.supply = world["supply"]
        self.kappa_true = world["kappa"][regime]
        self.onsets = world["onsets"]
        self.T = world["T"]
        self.t = 0
        self.R = cfg.R0
        self.S = 0.0
        self.C = 0.0
        self.s_known = cfg.s_known0
        self.kappa_model = cfg.kappa0
        # trajectories (post-action states)
        self.R_hist = np.empty(self.T)
        self.S_hist = np.empty(self.T)
        self.C_hist = np.empty(self.T)
        self.Ctrue_hist = np.empty(self.T)
        self.work_hist = np.empty(self.T)
        self.action_hist = np.empty(self.T, dtype=np.int8)
        self.sknown_hist = np.empty(self.T)

    # ---------------------------------------------------------------- controller view
    def feasible(self) -> np.ndarray:
        f = np.ones(N_ACTIONS, dtype=bool)
        f[SAMPLE] = self.R >= self.cfg.sample_floor - 1e-12
        f[UPDATE] = self.R >= self.cfg.update_floor - 1e-12
        return f

    def predicted_residuals(self) -> np.ndarray:
        """(N_ACTIONS, 3) signed one-step residuals r_hat_E, r_hat_S, r_hat_C under the model."""
        cfg = self.cfg
        cost = cfg.unit_cost * self.kappa_model * EFFORT + FIXED
        R_pred = np.clip(self.R + self.s_known - cost, 0.0, 1.0)
        rE = cfg.e_min - R_pred
        S_pred = np.minimum(1.0, self.S + cfg.s_rate) * np.ones(N_ACTIONS)
        S_pred[SAMPLE] = 0.0
        rS = S_pred - cfg.s_threshold
        C_pred = cfg.c_decay * self.C * np.ones(N_ACTIONS)
        C_pred[UPDATE] = 0.0
        rC = C_pred - cfg.c_threshold
        return np.column_stack([rE, rS, rC])

    # ---------------------------------------------------------------- dynamics
    def step(self, action: int) -> np.ndarray:
        """Apply one action; returns the realised signed residuals (E, S, C)."""
        cfg, t = self.cfg, self.t
        supply = self.supply[t]
        kt = self.kappa_true[t]
        fixed = FIXED[action]
        unit = cfg.unit_cost * kt
        effort_cmd = EFFORT[action]
        available = max(0.0, self.R + supply - fixed)
        effort = min(effort_cmd, available / unit) if effort_cmd > 0 else 0.0
        actual_spend = unit * effort
        predicted_spend = cfg.unit_cost * self.kappa_model * effort_cmd
        self.R = float(np.clip(self.R + supply - fixed - actual_spend, 0.0, 1.0))
        if action == SAMPLE:
            self.S = 0.0
            self.s_known = supply
        else:
            self.S = min(1.0, self.S + cfg.s_rate)
        if action == UPDATE:
            self.C = 0.0
            self.kappa_model = kt
        else:
            err = min(1.0, abs(actual_spend - predicted_spend) / predicted_spend) if predicted_spend > 0 else 0.0
            self.C = cfg.c_decay * self.C + (1.0 - cfg.c_decay) * err
        self.R_hist[t] = self.R
        self.S_hist[t] = self.S
        self.C_hist[t] = self.C
        self.Ctrue_hist[t] = min(1.0, abs(self.kappa_model - kt) / cfg.kappa0)
        self.work_hist[t] = effort
        self.action_hist[t] = action
        self.sknown_hist[t] = self.s_known
        self.t += 1
        return np.array([cfg.e_min - self.R, self.S - cfg.s_threshold, self.C - cfg.c_threshold])

    def summary(self) -> dict:
        cfg = self.cfg
        T = self.T
        post = np.zeros(T, dtype=bool)
        if self.onsets.size:
            post[self.onsets[0] :] = True
        positive = self.supply > 0
        absorbing = post & positive & (self.R_hist <= 1e-12) & (self.sknown_hist <= 1e-12)
        counts = np.bincount(self.action_hist, minlength=N_ACTIONS) / T
        return {
            "work": float(self.work_hist.mean()),
            "E_depth": float(np.maximum(0.0, cfg.e_min - self.R_hist).mean()),
            "S_depth": float(np.maximum(0.0, self.S_hist - cfg.s_threshold).mean()),
            "C_depth": float(np.maximum(0.0, self.C_hist - cfg.c_threshold).mean()),
            "C_true": float(self.Ctrue_hist.mean()),
            "absorbing_fraction": float(absorbing.sum() / max(1, (post & positive).sum())),
            "absorbing_ticks": int(absorbing.sum()),
            "zero_reserve_fraction": float((self.R_hist <= 1e-12).mean()),
            "action_freq": {ACTION_NAMES[i]: float(counts[i]) for i in range(N_ACTIONS)},
            "actions": self.action_hist.copy(),
        }


def run_plant(world: dict, regime: str, controller, cfg: PlantConfig = PlantConfig()) -> dict:
    """Drive a controller (reset(); act(plant) -> action; observe(residuals)) through one run."""
    plant = Plant(world, regime, cfg)
    controller.reset()
    while plant.t < plant.T:
        a = int(controller.act(plant))
        realised = plant.step(a)
        controller.observe(realised)
    return plant.summary()


# --------------------------------------------------------------------------- batched plant
class BatchedFixedPlant:
    """Simulate many constant-weight controllers at once on one world/regime.

    Weights W (K, 3) define action = argmin_a [-V effort(a) + sum_i W_i r_hat_i(a)] with
    signed (or hinged) predicted residuals, exactly the FIXED rule of the matched
    experiment; the state of every candidate evolves independently in numpy arrays so the
    1,000-candidate coarse search runs in one pass per training seed.
    """

    def __init__(self, world: dict, regime: str, weights: np.ndarray, V: float = 1.0, hinge: bool = False, cfg: PlantConfig = PlantConfig()):
        self.cfg = cfg
        self.supply = world["supply"]
        self.kappa_true = world["kappa"][regime]
        self.T = world["T"]
        self.W = np.asarray(weights, dtype=float)
        self.K = self.W.shape[0]
        self.V, self.hinge = V, hinge

    def run(self) -> dict:
        cfg, K, T = self.cfg, self.K, self.T
        R = np.full(K, cfg.R0)
        S = np.zeros(K)
        C = np.zeros(K)
        s_known = np.full(K, cfg.s_known0)
        kappa_model = np.full(K, cfg.kappa0)
        work_sum = np.zeros(K)
        E_depth = np.zeros(K)
        S_depth = np.zeros(K)
        C_depth = np.zeros(K)
        Ctrue_sum = np.zeros(K)
        idx = np.arange(K)
        unit0 = cfg.unit_cost
        for t in range(T):
            supply = self.supply[t]
            kt = self.kappa_true[t]
            # predicted residuals (K, A)
            cost = unit0 * kappa_model[:, None] * EFFORT[None, :] + FIXED[None, :]
            R_pred = np.clip(R[:, None] + s_known[:, None] - cost, 0.0, 1.0)
            rE = cfg.e_min - R_pred
            S_pred = np.repeat(np.minimum(1.0, S + cfg.s_rate)[:, None], N_ACTIONS, axis=1)
            S_pred[:, SAMPLE] = 0.0
            rS = S_pred - cfg.s_threshold
            C_pred = np.repeat((cfg.c_decay * C)[:, None], N_ACTIONS, axis=1)
            C_pred[:, UPDATE] = 0.0
            rC = C_pred - cfg.c_threshold
            if self.hinge:
                rE, rS, rC = np.maximum(rE, 0), np.maximum(rS, 0), np.maximum(rC, 0)
            score = -self.V * EFFORT[None, :] + self.W[:, 0:1] * rE + self.W[:, 1:2] * rS + self.W[:, 2:3] * rC
            infeasible = np.zeros((K, N_ACTIONS), dtype=bool)
            infeasible[:, SAMPLE] = R < cfg.sample_floor - 1e-12
            infeasible[:, UPDATE] = R < cfg.update_floor - 1e-12
            score = np.where(infeasible, np.inf, score)
            a = np.argmin(score, axis=1)  # first minimum on ties
            fixed = FIXED[a]
            unit = unit0 * kt
            effort_cmd = EFFORT[a]
            available = np.maximum(0.0, R + supply - fixed)
            effort = np.where(effort_cmd > 0, np.minimum(effort_cmd, available / unit), 0.0)
            actual = unit * effort
            predicted = unit0 * kappa_model * effort_cmd
            R = np.clip(R + supply - fixed - actual, 0.0, 1.0)
            is_sample = a == SAMPLE
            is_update = a == UPDATE
            S = np.where(is_sample, 0.0, np.minimum(1.0, S + cfg.s_rate))
            s_known = np.where(is_sample, supply, s_known)
            err = np.where(predicted > 0, np.minimum(1.0, np.abs(actual - predicted) / np.where(predicted > 0, predicted, 1.0)), 0.0)
            C = np.where(is_update, 0.0, cfg.c_decay * C + (1.0 - cfg.c_decay) * err)
            kappa_model = np.where(is_update, kt, kappa_model)
            work_sum += effort
            E_depth += np.maximum(0.0, cfg.e_min - R)
            S_depth += np.maximum(0.0, S - cfg.s_threshold)
            C_depth += np.maximum(0.0, C - cfg.c_threshold)
            Ctrue_sum += np.minimum(1.0, np.abs(kappa_model - kt) / cfg.kappa0)
        del idx
        return {
            "work": work_sum / T,
            "E_depth": E_depth / T,
            "S_depth": S_depth / T,
            "C_depth": C_depth / T,
            "C_true": Ctrue_sum / T,
        }
