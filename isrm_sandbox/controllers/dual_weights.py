"""Controllers for the reserve-and-supply plant (T08), written from the written protocol of
D:/ISRM Dual Weights (README.md, V3_PROTOCOL.md, V3_REPORT.md).

ADAPTIVE ("dual weights"):   Q_i <- max(0, Q_i + r_i) on the realised signed threshold
                             residuals; action = argmin_a [-V effort(a) + sum_i Q_i r_hat_i(a)]
                             over the feasible actions, V = 1, first minimum on ties.
NO_C:                        the same with the consistency queue held at zero.
FIXED:                       constant weights (alpha, beta, gamma) in place of the queues.
Drift-plus-penalty twin:     the textbook Lyapunov rule (Neely): minimise
                             V penalty(a) + sum_i Q_i [A_i(a) - mu_i(a)] with penalty = -effort
                             and the predicted net arrival A - mu = r_hat; virtual queue
                             Q_i <- max(0, Q_i + realised net arrival). Written separately so
                             that the claim "dual weights is this rule" is tested, not assumed.

Matching (V3_PROTOCOL.md): fit (alpha, beta, gamma) in [0, 64]^3 to ADAPTIVE's R2 depth
targets on the training worlds: 1,000 coarse triples from {0, .25, .5, 1, 2, 4, 8, 16, 32,
64}^3, then two refinement rounds, each crossing the three best candidates so far with
five multiplicative factors per coordinate (2^{-.5,-.25,0,.25,.5}, then 2^{-.25,...,.25}),
clipped to [0, 64], rounded to 12 decimals, deduplicated in generation order. Per-meter
tolerance max(0.10 target, 0.001 threshold); loss = worst absolute mismatch divided by its
tolerance; ties by the mean normalised mismatch, then candidate order. Loss <= 1 is a match.
"""
from __future__ import annotations

import numpy as np

from ..environments import reserve_supply_plant as RP

COARSE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
REFINEMENT_FACTORS = (
    tuple(2.0**e for e in (-0.5, -0.25, 0.0, 0.25, 0.5)),
    tuple(2.0**e for e in (-0.25, -0.125, 0.0, 0.125, 0.25)),
)
METERS = ("E_depth", "S_depth", "C_depth")
THRESHOLDS = (RP.THRESHOLDS["E"], RP.THRESHOLDS["S"], RP.THRESHOLDS["C"])


class VirtualQueueController:
    """ADAPTIVE (and NO_C when no_c=True)."""

    name = "adaptive_queues"

    def __init__(self, V: float = 1.0, no_c: bool = False, hinge: bool = False):
        self.V, self.no_c, self.hinge = V, no_c, hinge
        self.Q = np.zeros(3)

    def reset(self) -> None:
        self.Q = np.zeros(3)

    def weights(self) -> np.ndarray:
        return self.Q

    def act(self, plant: RP.Plant) -> int:
        rhat = plant.predicted_residuals()
        if self.hinge:
            rhat = np.maximum(rhat, 0.0)
        score = -self.V * RP.EFFORT + rhat @ self.weights()
        score = np.where(plant.feasible(), score, np.inf)
        return int(np.argmin(score))

    def observe(self, realised: np.ndarray) -> None:
        self.Q = np.maximum(0.0, self.Q + realised)
        if self.no_c:
            self.Q[2] = 0.0


class FixedWeightController(VirtualQueueController):
    name = "fixed_matched"

    def __init__(self, weights, V: float = 1.0, hinge: bool = False):
        super().__init__(V=V, hinge=hinge)
        self.W = np.asarray(weights, dtype=float)

    def weights(self) -> np.ndarray:
        return self.W

    def observe(self, realised: np.ndarray) -> None:
        return None


class DriftPlusPenalty:
    """Textbook drift-plus-penalty with virtual queues (independent formulation)."""

    name = "dpp_twin"

    def __init__(self, V: float = 1.0):
        self.V = V
        self.backlog = np.zeros(3)

    def reset(self) -> None:
        self.backlog = np.zeros(3)

    def act(self, plant: RP.Plant) -> int:
        net_arrival = plant.predicted_residuals()  # A_i(a) - mu_i(a) for each constraint i
        penalty = -RP.EFFORT  # the penalty to minimise is negative work
        drift_plus_penalty = self.V * penalty + net_arrival @ self.backlog
        drift_plus_penalty = np.where(plant.feasible(), drift_plus_penalty, np.inf)
        return int(np.argmin(drift_plus_penalty))

    def observe(self, realised_net_arrival: np.ndarray) -> None:
        self.backlog = np.maximum(0.0, self.backlog + realised_net_arrival)


# --------------------------------------------------------------------------- matching
def tolerance(target: np.ndarray) -> np.ndarray:
    return np.maximum(0.10 * np.asarray(target), 0.001 * np.asarray(THRESHOLDS))


def match_loss(depths: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Worst and mean normalised mismatch for depths (K, 3) against target (3,)."""
    norm = np.abs(depths - target) / tolerance(target)
    return norm.max(axis=1), norm.mean(axis=1)


def _dedupe(cands: np.ndarray, seen: set) -> np.ndarray:
    out = []
    for c in np.round(np.clip(cands, 0.0, 64.0), 12):
        key = tuple(c.tolist())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return np.array(out) if out else np.zeros((0, 3))


def evaluate_candidates(cands: np.ndarray, worlds: list[dict], regime: str, V: float, hinge: bool, cfg: RP.PlantConfig) -> np.ndarray:
    """Mean (E, S, C) depths over the training worlds for every candidate triple."""
    depths = np.zeros((len(cands), 3))
    for w in worlds:
        out = RP.BatchedFixedPlant(w, regime, cands, V=V, hinge=hinge, cfg=cfg).run()
        depths += np.column_stack([out[m] for m in METERS])
    return depths / len(worlds)


def match_fixed_weights(worlds: list[dict], target: np.ndarray, regime: str = "R2", V: float = 1.0, hinge: bool = False, cfg: RP.PlantConfig = RP.PlantConfig()) -> dict:
    """Bounded coarse-plus-refinement search for constant weights matching `target` depths."""
    seen: set = set()
    g = np.array(COARSE_GRID)
    coarse = np.array([(a, b, c) for a in g for b in g for c in g])
    cands = _dedupe(coarse, seen)
    all_cands = [cands]
    all_loss = [None]
    all_mean = [None]
    depths = evaluate_candidates(cands, worlds, regime, V, hinge, cfg)
    worst, mean = match_loss(depths, target)
    all_loss[0], all_mean[0] = worst, mean
    all_depths = [depths]
    for factors in REFINEMENT_FACTORS:
        C = np.concatenate(all_cands)
        L = np.concatenate(all_loss)
        M = np.concatenate(all_mean)
        order = np.lexsort((np.arange(len(C)), M, L))
        best3 = C[order[:3]]
        f = np.array(factors)
        new = np.array([[c[0] * fa, c[1] * fb, c[2] * fc] for c in best3 for fa in f for fb in f for fc in f])
        new = _dedupe(new, seen)
        if len(new) == 0:
            continue
        d = evaluate_candidates(new, worlds, regime, V, hinge, cfg)
        w, m = match_loss(d, target)
        all_cands.append(new)
        all_loss.append(w)
        all_mean.append(m)
        all_depths.append(d)
    C = np.concatenate(all_cands)
    L = np.concatenate(all_loss)
    M = np.concatenate(all_mean)
    D = np.concatenate(all_depths)
    order = np.lexsort((np.arange(len(C)), M, L))
    b = order[0]
    return {
        "weights": C[b].tolist(),
        "worst_normalised_mismatch": float(L[b]),
        "mean_normalised_mismatch": float(M[b]),
        "matched": bool(L[b] <= 1.0),
        "fitted_depths": D[b].tolist(),
        "target_depths": np.asarray(target).tolist(),
        "tolerance": tolerance(target).tolist(),
        "n_candidates": int(len(C)),
        "n_matching_candidates": int((L <= 1.0).sum()),
    }
