"""Frozen expert bank over a piecewise-stationary stream (the expert-switching setting).

Independent re-creation of the world used by the expert-switching ISRM studies:
K experts, each competent on exactly one latent regime, a hidden Markov regime
process with mean persistence p, labels released d batches late, and a
controller that each batch either serves one expert (1 forward per input) or
buys a diagnostic scan of the whole bank (K forwards per input). The controller's
only job is to notice regime changes and serve the right expert; the claim under
test is about *when* it buys scans.

Everything is drawn from one numpy Generator so every arm sees the same world.
Expert outputs on every batch are cached up front (errs[k, t]) so an episode is
pure bookkeeping, as in the original "cached logits" runs.

Regime rule k:  y = 1[ x . w_k + amp * sin(1.5 * x . v_k) > 0 ],  x ~ N(0, I_d),
with orthogonal w_k and random unit v_k. Experts are logistic regressions fitted
by full-batch gradient descent on their own regime, so they are good but
imperfect (the sin term is not linearly separable), which gives realistic
in-regime error noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FrozenBankConfig:
    n_regimes: int = 8
    dim: int = 8
    batch_size: int = 8
    persistence: int = 320
    delay: int = 2
    horizon: int = 4096
    nonlinearity: float = 0.4
    train_samples: int = 2048
    train_steps: int = 300
    lr: float = 0.5
    calib_samples: int = 4096


def _rule(X: np.ndarray, w: np.ndarray, v: np.ndarray, amp: float) -> np.ndarray:
    return (X @ w + amp * np.sin(1.5 * (X @ v)) > 0).astype(np.int8)


def _fit_logistic(X: np.ndarray, y: np.ndarray, steps: int, lr: float) -> tuple[np.ndarray, float]:
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        z = np.clip(X @ w + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        w -= lr * (X.T @ g / len(y))
        b -= lr * float(np.mean(g))
    return w, b


def _predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return (X @ w + b > 0).astype(np.int8)


def make_world(rng: np.random.Generator, cfg: FrozenBankConfig) -> dict:
    K, d, T, B = cfg.n_regimes, cfg.dim, cfg.horizon, cfg.batch_size
    if K <= d:
        Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        W = Q[:, :K].T.copy()
    else:  # more regimes than dimensions: random unit vectors
        W = rng.normal(size=(K, d))
        W /= np.linalg.norm(W, axis=1, keepdims=True)
    V = rng.normal(size=(K, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    experts = []
    own_err = np.empty(K)
    for k in range(K):
        Xtr = rng.normal(size=(cfg.train_samples, d))
        ytr = _rule(Xtr, W[k], V[k], cfg.nonlinearity)
        w, b = _fit_logistic(Xtr, ytr, cfg.train_steps, cfg.lr)
        experts.append((w, b))
        Xc = rng.normal(size=(cfg.calib_samples, d))
        yc = _rule(Xc, W[k], V[k], cfg.nonlinearity)
        own_err[k] = float(np.mean(_predict(Xc, w, b) != yc))

    # hidden Markov regime process: with probability 1/p jump to another regime
    schedule = np.empty(T, dtype=int)
    k = int(rng.integers(K))
    changes = 0
    for t in range(T):
        if t > 0 and rng.random() < 1.0 / cfg.persistence:
            k = int((k + rng.integers(1, K)) % K)
            changes += 1
        schedule[t] = k

    X = rng.normal(size=(T, B, d))
    y = np.empty((T, B), dtype=np.int8)
    for t in range(T):
        y[t] = _rule(X[t], W[schedule[t]], V[schedule[t]], cfg.nonlinearity)

    errs = np.empty((K, T))
    flatX = X.reshape(T * B, d)
    for k, (w, b) in enumerate(experts):
        pred = _predict(flatX, w, b).reshape(T, B)
        errs[k] = np.mean(pred != y, axis=1)

    return {
        "errs": errs,
        "schedule": schedule,
        "regime_changes": changes,
        "own_err": own_err,
        "b_ref": float(own_err.mean()),
        "K": K,
        "T": T,
    }


class BankObserver:
    """Discounted per-expert error observer with a 0.5 prior of weight 1."""

    def __init__(self, K: int, gamma: float):
        self.K, self.gamma = K, gamma
        self.sums = np.zeros(K)
        self.weights = np.zeros(K)

    def decay(self) -> None:
        self.sums *= self.gamma
        self.weights *= self.gamma

    def receive(self, e: int, err: float, age: int) -> None:
        w = self.gamma**age
        self.sums[e] += w * err
        self.weights[e] += w

    def scores(self) -> np.ndarray:
        return (self.sums + 0.5) / (self.weights + 1.0)

    def serve(self) -> int:
        return int(np.argmin(self.scores()))  # lowest index on ties


class ScanPolicy:
    """Decides when to buy a whole-bank diagnostic. Subclasses override the hooks."""

    name = "policy"
    gamma = 0.8  # observer discount owned by the policy so grids can vary it

    def reset(self, K: int, b_ref: float, delay: int, horizon: int) -> None:
        self.K, self.b, self.d, self.T = K, b_ref, delay, horizon

    def on_receipt(self, t: int, s: int, e: int, err: float, eligible: bool) -> None:
        return None

    def want_scan(self, t: int, last_diag: int, served: int) -> bool:
        return False

    def on_scan(self, t: int) -> None:
        return None


class CUSUMEventPolicy(ScanPolicy):
    """The event controller of the expert-switching studies (re-implemented from spec).

    Per-expert one-sided CUSUM on receipts of the expert that was served:
    cusum_e = max(0, cusum_e + gamma^d (err_e(s) - b)). Request a scan when the
    currently served expert's CUSUM exceeds theta and at least delay+1 batches
    have passed since the last scan. Scans reset every CUSUM.
    """

    name = "event_cusum"

    def __init__(self, theta: float = 0.5, gamma: float = 0.8):
        self.theta, self.gamma = theta, gamma

    def reset(self, K, b_ref, delay, horizon):
        super().reset(K, b_ref, delay, horizon)
        self.cusum = np.zeros(K)

    def on_receipt(self, t, s, e, err, eligible):
        if eligible:
            self.cusum[e] = max(0.0, self.cusum[e] + (self.gamma**self.d) * (err - self.b))

    def want_scan(self, t, last_diag, served):
        return self.cusum[served] > self.theta and (t - last_diag) >= self.d + 1

    def on_scan(self, t):
        self.cusum[:] = 0.0


class EWMAScanPolicy(ScanPolicy):
    """The conventional adaptive scanner: EWMA of the served expert's error.

    eps <- beta eps + (1-beta) err on eligible receipts; scan when eps > b + margin
    or when max_gap batches have passed since the last scan; eps resets to b.
    """

    name = "adaptive_ewma"

    def __init__(self, beta: float = 0.95, margin: float = 0.03, max_gap: int = 64, gamma: float = 0.8, min_gap: int = 1):
        self.beta, self.margin, self.max_gap, self.gamma, self.min_gap = beta, margin, max_gap, gamma, min_gap

    def reset(self, K, b_ref, delay, horizon):
        super().reset(K, b_ref, delay, horizon)
        self.eps = b_ref

    def on_receipt(self, t, s, e, err, eligible):
        if eligible:
            self.eps = self.beta * self.eps + (1.0 - self.beta) * err

    def want_scan(self, t, last_diag, served):
        gap = t - last_diag
        return gap >= self.min_gap and (self.eps > self.b + self.margin or gap >= self.max_gap)

    def on_scan(self, t):
        self.eps = self.b


class PeriodicScanPolicy(ScanPolicy):
    name = "periodic"

    def __init__(self, period: int, gamma: float = 0.8):
        self.period, self.gamma = period, gamma

    def want_scan(self, t, last_diag, served):
        return (t - last_diag) >= self.period


class DenseScanPolicy(ScanPolicy):
    """Scan every batch: the accuracy ceiling for observer-based serving."""

    name = "dense"

    def __init__(self, gamma: float = 0.8):
        self.gamma = gamma

    def want_scan(self, t, last_diag, served):
        return True


class NeverScanPolicy(ScanPolicy):
    """Only the mandatory initial scan: the persistence floor."""

    name = "never"


class OracleScanPolicy(ScanPolicy):
    """Scans exactly when a regime change becomes observable (needs the schedule)."""

    name = "oracle"

    def __init__(self, schedule: np.ndarray, gamma: float = 0.8):
        self.schedule, self.gamma = schedule, gamma

    def want_scan(self, t, last_diag, served):
        s = t - self.d
        return s > 0 and self.schedule[s] != self.schedule[s - 1] and last_diag < s


def run_episode(world: dict, cfg: FrozenBankConfig, policy: ScanPolicy) -> dict:
    errs, K, T, d = world["errs"], world["K"], world["T"], cfg.delay
    obs = BankObserver(K, policy.gamma)
    policy.reset(K, world["b_ref"], d, T)
    executed: list[list[int] | None] = [None] * T
    served = np.zeros(T, dtype=int)
    correct = np.zeros(T)
    forwards = 0
    scans = 0
    last_diag = -(10**9)
    all_experts = list(range(K))

    for t in range(T):
        obs.decay()
        s = t - d
        if s >= 0:
            eligible = s >= last_diag
            for e in executed[s]:  # type: ignore[union-attr]
                obs.receive(e, errs[e, s], d)
            policy.on_receipt(t, s, int(served[s]), errs[served[s], s], eligible)
        c = obs.serve()
        scan = t == 0 or policy.want_scan(t, last_diag, c)
        if scan and t > 0 and t + d >= T:
            scan = False  # receipts would never arrive
        if scan:
            executed[t] = all_experts
            last_diag = t
            policy.on_scan(t)
            forwards += K
            scans += 1
        else:
            executed[t] = [c]
            forwards += 1
        served[t] = c
        correct[t] = 1.0 - errs[c, t]

    q = T // 4
    return {
        "accuracy": float(correct.mean()),
        "final_quarter_accuracy": float(correct[-q:].mean()) if q else float(correct.mean()),
        "work": forwards / T,  # forwards per input
        "scans": scans,
        "regime_changes": int(world["regime_changes"]),
    }
