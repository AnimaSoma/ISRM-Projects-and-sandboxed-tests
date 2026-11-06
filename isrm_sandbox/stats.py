"""Statistics for paired arm comparisons and verdict assignment.

All randomness comes from a numpy Generator supplied by the caller (seeded by
the runner from the spec hash), so bootstrap intervals are reproducible.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass
class Contrast:
    treatment: str
    comparator: str
    n: int
    mean_treatment: float
    mean_comparator: float
    mean_diff: float
    ci_low: float
    ci_high: float
    alpha: float
    p_sign_flip: float
    wins: int  # replicates where treatment beat comparator
    losses: int
    ties: int

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap(
    x: Sequence[float],
    y: Sequence[float],
    rng: np.random.Generator,
    n_boot: int = 20000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean of (x - y) with a percentile bootstrap interval over paired replicates."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    if d.ndim != 1 or d.size < 2:
        raise ValueError("need at least two paired replicates")
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def sign_flip_p(
    x: Sequence[float],
    y: Sequence[float],
    rng: np.random.Generator,
    n_perm: int = 20000,
    alternative: str = "two-sided",
) -> float:
    """Sign-flip permutation p-value for a paired mean difference."""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    observed = d.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = (signs * d).mean(axis=1)
    if alternative == "greater":
        p = (np.sum(perm >= observed) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(perm <= observed) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(perm) >= abs(observed)) + 1) / (n_perm + 1)
    return float(min(1.0, p))


def holm(pvals: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted.tolist()


def contrast(
    name_t: str,
    name_c: str,
    x: Sequence[float],
    y: Sequence[float],
    rng: np.random.Generator,
    alpha: float,
    higher_is_better: bool,
    n_boot: int = 20000,
) -> Contrast:
    """Paired contrast expressed so that a positive mean_diff favours the treatment."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    if not higher_is_better:
        xs, ys = -xs, -ys
    mean_diff, lo, hi = paired_bootstrap(xs, ys, rng, n_boot=n_boot, alpha=alpha)
    p = sign_flip_p(xs, ys, rng, alternative="two-sided")
    d = xs - ys
    return Contrast(
        treatment=name_t,
        comparator=name_c,
        n=int(d.size),
        mean_treatment=float(np.mean(x)),
        mean_comparator=float(np.mean(y)),
        mean_diff=mean_diff,
        ci_low=lo,
        ci_high=hi,
        alpha=alpha,
        p_sign_flip=p,
        wins=int(np.sum(d > 0)),
        losses=int(np.sum(d < 0)),
        ties=int(np.sum(d == 0)),
    )


def verdict(
    c: Contrast,
    direction: str,
    threshold: float,
    labels: dict[str, str],
    equivalence_margin: float | None = None,
) -> str:
    """Map a contrast (oriented so + favours the treatment) to a status label.

    direction 'greater': the treatment is claimed to beat the comparator by at
    least `threshold`. 'less' is the same claim stated on a lower-is-better
    metric and is already handled by the orientation in `contrast`, so it is
    treated identically here. 'equivalence' claims the arms differ by less than
    `equivalence_margin` (TOST: the whole interval lies inside the margin).
    """
    lo, hi = c.ci_low, c.ci_high
    if direction == "equivalence":
        m = float(equivalence_margin or 0.0)
        if -m < lo and hi < m:
            return labels["equivalent"]
        if lo > 0:
            return labels["none"]
        if hi < 0:
            return labels["harmful"]
        return labels["inconclusive"]
    if lo >= threshold:
        return labels["supported"]
    if hi < 0:
        return labels["harmful"]
    if lo > 0:
        return labels["below_threshold"]
    if equivalence_margin is not None and -equivalence_margin < lo and hi < equivalence_margin:
        return labels["equivalent"]
    width = hi - lo
    if threshold > 0 and width > 4 * threshold:
        return labels["inconclusive"]
    return labels["none"]
