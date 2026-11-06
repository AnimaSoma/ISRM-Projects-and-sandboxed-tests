"""Matched conventional comparators for event-triggered supervision.

These are the controllers an ISRM supervisor must beat to show a distinctive
effect. They see exactly the same signal (a per-step prediction-error or loss
value) and are allowed the same actions. Each exposes

    step(error: float) -> bool      # True = trigger (switch / revise / escalate)
    reset() -> None

so that an environment can drive any of them interchangeably with an ISRM arm.
All are classical change-detection or hysteresis rules; none uses ISRM concepts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class Trigger:
    """Interface shared by every comparator and by the ISRM arms."""

    name: str = "trigger"

    def step(self, error: float) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class FixedThreshold(Trigger):
    """Trigger when the instantaneous error exceeds a constant threshold."""

    threshold: float
    name: str = "fixed_threshold"

    def step(self, error: float) -> bool:
        return error > self.threshold

    def reset(self) -> None:
        return None


@dataclass
class HysteresisBand(Trigger):
    """Two-threshold Schmitt trigger: arm above `high`, disarm below `low`.

    Fires once on each upward crossing. This is the classic governor any
    threshold-with-dwell rule reduces to.
    """

    low: float
    high: float
    name: str = "hysteresis_band"
    _armed: bool = field(default=False, init=False)

    def step(self, error: float) -> bool:
        if not self._armed and error > self.high:
            self._armed = True
            return True
        if self._armed and error < self.low:
            self._armed = False
        return False

    def reset(self) -> None:
        self._armed = False


@dataclass
class EWMAThreshold(Trigger):
    """Trigger when an exponentially weighted mean error exceeds a threshold.

    `half_life` is in steps. Resets the average after firing so repeated
    triggers need fresh evidence.
    """

    threshold: float
    half_life: float = 8.0
    name: str = "ewma_threshold"
    _m: float = field(default=0.0, init=False)
    _initialised: bool = field(default=False, init=False)

    @property
    def lam(self) -> float:
        return 1.0 - 0.5 ** (1.0 / max(self.half_life, 1e-9))

    def step(self, error: float) -> bool:
        if not self._initialised:
            self._m, self._initialised = error, True
        else:
            self._m += self.lam * (error - self._m)
        if self._m > self.threshold:
            self._initialised = False
            return True
        return False

    def reset(self) -> None:
        self._m, self._initialised = 0.0, False


@dataclass
class CUSUM(Trigger):
    """One-sided cumulative-sum detector for an upward shift in error.

    S_t = max(0, S_{t-1} + (e_t - target - drift)); trigger when S_t > h.
    `target` is the in-control mean error (estimated from a warm-up if None),
    `drift` the allowance k, `h` the decision interval. Resets after firing.
    """

    h: float
    drift: float = 0.0
    target: float | None = None
    warmup: int = 20
    name: str = "cusum"
    _s: float = field(default=0.0, init=False)
    _buf: list = field(default_factory=list, init=False)
    _mu: float | None = field(default=None, init=False)

    def step(self, error: float) -> bool:
        if self._mu is None:
            if self.target is not None:
                self._mu = self.target
            else:
                self._buf.append(error)
                if len(self._buf) < self.warmup:
                    return False
                self._mu = float(np.mean(self._buf))
        self._s = max(0.0, self._s + (error - self._mu - self.drift))
        if self._s > self.h:
            self._s = 0.0
            return True
        return False

    def reset(self) -> None:
        self._s = 0.0
        self._buf = []
        self._mu = None if self.target is None else self.target


@dataclass
class PageHinkley(Trigger):
    """Page-Hinkley test for an increase in the mean error.

    m_t = sum_{i<=t} (e_i - mean_t - delta); PH_t = m_t - min_{i<=t} m_i; trigger
    when PH_t > lambda. Standard drift detector in streaming learning.
    """

    lam: float
    delta: float = 0.005
    name: str = "page_hinkley"
    _n: int = field(default=0, init=False)
    _mean: float = field(default=0.0, init=False)
    _m: float = field(default=0.0, init=False)
    _min_m: float = field(default=0.0, init=False)

    def step(self, error: float) -> bool:
        self._n += 1
        self._mean += (error - self._mean) / self._n
        self._m += error - self._mean - self.delta
        self._min_m = min(self._min_m, self._m)
        if self._m - self._min_m > self.lam:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self._n, self._mean, self._m, self._min_m = 0, 0.0, 0.0, 0.0


@dataclass
class PeriodicTrigger(Trigger):
    """Fires every `period` steps regardless of error: the fixed-schedule control."""

    period: int
    name: str = "periodic"
    _t: int = field(default=0, init=False)

    def step(self, error: float) -> bool:
        self._t += 1
        if self._t >= self.period:
            self._t = 0
            return True
        return False

    def reset(self) -> None:
        self._t = 0


@dataclass
class NeverTrigger(Trigger):
    """Never fires: the persistence control."""

    name: str = "never"

    def step(self, error: float) -> bool:
        return False

    def reset(self) -> None:
        return None


def tune_threshold(
    make: "callable",
    grid: np.ndarray,
    objective: "callable",
) -> tuple[float, float]:
    """Pick the grid value maximising `objective(trigger)`; ties go to the smaller value.

    `make(value)` builds the comparator; `objective` must evaluate it on
    development data only. The runner's paired design keeps evaluation seeds
    disjoint, so tuning here never sees evaluation worlds.
    """
    best_v, best_o = None, -np.inf
    for v in grid:
        o = objective(make(float(v)))
        if o > best_o:
            best_v, best_o = float(v), float(o)
    return best_v, best_o
