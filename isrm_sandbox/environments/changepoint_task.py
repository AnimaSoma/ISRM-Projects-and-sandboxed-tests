"""Change-point learning task for the decision-debt test (manuscript section 5.1).

Two candidate models, A and B. A hidden rule says which one currently predicts the
binary observation; the correct model's prediction is emitted with a fixed
accuracy (0.62 in the manuscript), so single observations are weak evidence and
the discrepancy must be inferred over several trials.

Protocol (where the manuscript is silent the original simulation's choices are
followed, as documented in the theory audit): every episode begins with the
adopted model *wrong* (rule = 1, adopted model = 0) and a maximally uncertain
belief, so a reliable, persistent discrepancy exists from the first trial. The
rule may additionally flip with `world_hazard` (0 in the original protocol; the
controller's filter assumes `filter_hazard` regardless).

A participant is a bundle of pre-drawn random streams: preparation episodes
(6 x 40 trials) and a test episode (40 trials). Both preparation conditions
(resolved: capacity 1.0; deferred: capacity 0.35) and every controller receive the
same streams, so any difference at test is caused by the controller's carried-over
state, exactly as the manuscript's design intends.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskConfig:
    accuracy: float = 0.62
    world_hazard: float = 0.0
    prep_episodes: int = 6
    prep_trials: int = 40
    test_trials: int = 40
    initial_rule: int = 1  # adopted model is 0, so the model starts wrong


@dataclass
class Episode:
    rule: np.ndarray  # correct model id per trial (0 = A predicts o=1, 1 = B predicts o=0)
    obs: np.ndarray  # observation per trial in {0, 1}


def _episode(rng: np.random.Generator, n: int, cfg: TaskConfig) -> Episode:
    rule = np.empty(n, dtype=int)
    k = cfg.initial_rule
    for t in range(n):
        if t > 0 and cfg.world_hazard > 0 and rng.random() < cfg.world_hazard:
            k = 1 - k
        rule[t] = k
    correct_pred = np.where(rule == 0, 1, 0)
    hit = rng.random(n) < cfg.accuracy
    obs = np.where(hit, correct_pred, 1 - correct_pred)
    return Episode(rule=rule, obs=obs.astype(int))


def make_participant(rng: np.random.Generator, cfg: TaskConfig) -> dict:
    prep = [_episode(rng, cfg.prep_trials, cfg) for _ in range(cfg.prep_episodes)]
    test = _episode(rng, cfg.test_trials, cfg)
    return {"prep": prep, "test": test}


def model_prediction(model: int) -> int:
    """Model A (0) predicts observation 1, model B (1) predicts observation 0."""
    return 1 if model == 0 else 0
