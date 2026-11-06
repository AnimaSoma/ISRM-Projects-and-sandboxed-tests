"""Piecewise-stationary classification stream with delayed feedback and an expert bank.

Independent re-creation of the setting used by the expert-switching ISRM studies
(regimes with a mean persistence, feedback delay, a bank of experts, a controller
that decides *when* to re-evaluate and switch). Nothing here is copied from those
projects; the generator, experts and accounting are the sandbox's own.

World
-----
* K latent regimes, each a linear-threshold rule y = 1[w_k . x + b_k + eps > 0]
  over x ~ N(0, I_d). Regime durations are geometric with mean `persistence`
  batches (or fixed when `fixed_durations`). The next regime is drawn uniformly
  among the others.
* Batches of `batch_size` points arrive at t = 0..horizon-1. Labels of batch t
  become available at t + delay.

Experts
-------
* A bank of `n_experts` logistic-regression experts (closed-form gradient steps).
* Work accounting (abstract units, no wall clock):
    EVAL_COST  per expert evaluated on one batch,
    TRAIN_COST per gradient step on one batch,
    PREDICT_COST per batch prediction with the active expert.

Controller protocol
-------------------
A controller is any object with `step(error) -> bool` (see controllers.baselines).
At each batch the environment:
  1. predicts the batch with the active expert (always paid);
  2. when the delayed labels for batch t-delay arrive, computes that batch's error
     for the active expert and feeds it to the controller;
  3. if the controller fires, runs the *shared* actuation policy: evaluate every
     expert on the latest labeled batch, switch to the best, and if even the
     best exceeds `retrain_error`, retrain the worst-ranked expert on the recent
     labeled window and switch to it.
  4. otherwise performs `maintenance_steps` cheap gradient steps on the active
     expert with the newly labeled batch.

Because the actuation policy is shared, two controllers differ only in *when*
they trigger, which is the quantity under test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EVAL_COST = 1.0
TRAIN_COST = 1.0
PREDICT_COST = 0.1


@dataclass
class StreamConfig:
    n_regimes: int = 4
    dim: int = 16
    persistence: int = 80
    delay: int = 2
    batch_size: int = 32
    horizon: int = 512
    label_noise: float = 0.05
    fixed_durations: bool = False
    regime_separation: float = 1.0  # scale of regime weight vectors


@dataclass
class BankConfig:
    n_experts: int = 4
    lr: float = 0.5
    l2: float = 1e-3
    maintenance_steps: int = 1
    retrain_steps: int = 10
    retrain_error: float = 0.25
    window: int = 4  # labeled batches used for retraining


class LogisticExpert:
    def __init__(self, dim: int, rng: np.random.Generator, lr: float, l2: float):
        self.w = rng.normal(0, 0.01, dim)
        self.b = 0.0
        self.lr, self.l2 = lr, l2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def error(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) != y))

    def sgd_step(self, X: np.ndarray, y: np.ndarray) -> None:
        p = self.predict_proba(X)
        g = p - y
        self.w -= self.lr * (X.T @ g / len(y) + self.l2 * self.w)
        self.b -= self.lr * float(np.mean(g))


@dataclass
class EpisodeResult:
    accuracy: float
    work: float
    switches: int
    triggers: int
    regime_changes: int
    final_quarter_accuracy: float
    per_batch_accuracy: np.ndarray = field(repr=False)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "work": self.work,
            "switches": self.switches,
            "triggers": self.triggers,
            "regime_changes": self.regime_changes,
            "final_quarter_accuracy": self.final_quarter_accuracy,
        }


def make_world(rng: np.random.Generator, cfg: StreamConfig) -> dict:
    """Draw the regime parameters, the regime schedule and all batches up front.

    Returning the whole world lets every arm see identical data (paired design).
    """
    W = rng.normal(0, cfg.regime_separation, (cfg.n_regimes, cfg.dim))
    B = rng.normal(0, 0.1, cfg.n_regimes)
    schedule = np.empty(cfg.horizon, dtype=int)
    t, k = 0, int(rng.integers(cfg.n_regimes))
    changes = 0
    while t < cfg.horizon:
        dur = cfg.persistence if cfg.fixed_durations else max(1, int(rng.geometric(1.0 / cfg.persistence)))
        end = min(cfg.horizon, t + dur)
        schedule[t:end] = k
        t = end
        if t < cfg.horizon:
            others = [j for j in range(cfg.n_regimes) if j != k]
            k = int(rng.choice(others))
            changes += 1
    X = rng.normal(0, 1, (cfg.horizon, cfg.batch_size, cfg.dim))
    eps = rng.normal(0, 1, (cfg.horizon, cfg.batch_size)) * 0.0
    logits = np.einsum("tbd,td->tb", X, W[schedule]) + B[schedule][:, None] + eps
    y = (logits > 0).astype(int)
    flip = rng.random((cfg.horizon, cfg.batch_size)) < cfg.label_noise
    y = np.where(flip, 1 - y, y)
    return {"X": X, "y": y, "schedule": schedule, "regime_changes": changes, "W": W, "B": B}


def run_episode(world: dict, cfg: StreamConfig, bank_cfg: BankConfig, controller, rng: np.random.Generator) -> EpisodeResult:
    """Drive one controller through a pre-drawn world. `rng` seeds only the expert init."""
    X, y, horizon = world["X"], world["y"], cfg.horizon
    experts = [LogisticExpert(cfg.dim, rng, bank_cfg.lr, bank_cfg.l2) for _ in range(bank_cfg.n_experts)]
    active = 0
    work = 0.0
    switches = triggers = 0
    correct = np.zeros(horizon)
    labeled: list[int] = []  # indices of batches whose labels have arrived
    controller.reset()

    for t in range(horizon):
        pred = experts[active].predict(X[t])
        correct[t] = float(np.mean(pred == y[t]))
        work += PREDICT_COST

        s = t - cfg.delay  # batch whose labels arrive now
        if s < 0:
            continue
        labeled.append(s)
        err = experts[active].error(X[s], y[s])
        work += EVAL_COST
        fired = controller.step(err)

        if fired:
            triggers += 1
            errs = [e.error(X[s], y[s]) for e in experts]
            work += EVAL_COST * len(experts)
            best = int(np.argmin(errs))
            if errs[best] > bank_cfg.retrain_error:
                worst = int(np.argmax(errs))
                recent = labeled[-bank_cfg.window :]
                Xr = np.concatenate([X[i] for i in recent])
                yr = np.concatenate([y[i] for i in recent])
                for _ in range(bank_cfg.retrain_steps):
                    experts[worst].sgd_step(Xr, yr)
                work += TRAIN_COST * bank_cfg.retrain_steps * len(recent)
                best = worst
            if best != active:
                switches += 1
                active = best
        else:
            for _ in range(bank_cfg.maintenance_steps):
                experts[active].sgd_step(X[s], y[s])
            work += TRAIN_COST * bank_cfg.maintenance_steps

    q = horizon // 4
    return EpisodeResult(
        accuracy=float(correct.mean()),
        work=work,
        switches=switches,
        triggers=triggers,
        regime_changes=int(world["regime_changes"]),
        final_quarter_accuracy=float(correct[-q:].mean()) if q else float(correct.mean()),
        per_batch_accuracy=correct,
    )


class OracleTrigger:
    """Fires exactly when the regime changes (after the feedback delay): the ceiling.

    Needs the schedule, so it is only usable inside the sandbox as a control arm.
    """

    name = "oracle"

    def __init__(self, schedule: np.ndarray, delay: int):
        self.schedule, self.delay = schedule, delay
        self._t = -1

    def reset(self) -> None:
        self._t = -1

    def step(self, error: float) -> bool:
        self._t += 1
        s = self._t  # index of the batch whose labels just arrived
        return s > 0 and self.schedule[s] != self.schedule[s - 1]
