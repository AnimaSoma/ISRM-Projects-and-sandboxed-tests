"""Protocol tests for the matched comparators: they must detect a genuine shift and
stay quiet on a stationary stream, otherwise an ISRM arm would be beating a broken
baseline rather than a conventional one.
"""
from __future__ import annotations

import numpy as np
import pytest

from isrm_sandbox.controllers import baselines as B

pytestmark = pytest.mark.protocol


def _stream(rng, n_pre=200, n_post=200, shift=1.0):
    pre = rng.normal(0.0, 0.3, n_pre)
    post = rng.normal(shift, 0.3, n_post)
    return np.concatenate([pre, post]), n_pre


@pytest.mark.parametrize(
    "make",
    [
        lambda: B.CUSUM(h=2.0, drift=0.1),
        lambda: B.PageHinkley(lam=3.0, delta=0.05),
        lambda: B.EWMAThreshold(threshold=0.6, half_life=6),
        lambda: B.HysteresisBand(low=0.4, high=0.8),
    ],
)
def test_detectors_fire_after_shift_and_rarely_before(make):
    rng = np.random.default_rng(11)
    errs, onset = _stream(rng)
    trig = make()
    fires = [t for t, e in enumerate(errs) if trig.step(float(e))]
    false_alarms = [t for t in fires if t < onset]
    detections = [t for t in fires if t >= onset]
    assert detections, f"{trig.name} never fired after the shift"
    assert detections[0] - onset < 40, f"{trig.name} detection delay too long: {detections[0] - onset}"
    assert len(false_alarms) <= 2, f"{trig.name} false alarms before onset: {false_alarms}"


def test_periodic_and_never():
    p = B.PeriodicTrigger(period=5)
    fires = [p.step(0.0) for _ in range(20)]
    assert sum(fires) == 4
    n = B.NeverTrigger()
    assert not any(n.step(10.0) for _ in range(10))


def test_reset_clears_state():
    c = B.CUSUM(h=1.0, drift=0.0, target=0.0)
    for _ in range(3):
        c.step(0.3)
    c.reset()
    assert c._s == 0.0


def test_tune_threshold_prefers_best_objective():
    grid = np.array([0.1, 0.5, 1.0])
    v, o = B.tune_threshold(lambda t: B.FixedThreshold(t), grid, lambda trig: -abs(trig.threshold - 0.5))
    assert v == 0.5 and o == 0.0
