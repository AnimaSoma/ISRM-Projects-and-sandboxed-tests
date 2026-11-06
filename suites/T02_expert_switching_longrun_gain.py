"""T02: does the CUSUM event controller's long-run accuracy gain over the
cost-selected adaptive scanner survive in an independent world, and does it
survive symmetric comparator selection?"""
from __future__ import annotations

from _expert_switching import evaluate


def run(spec, replicates, seed_for):
    return evaluate(spec, replicates, seed_for, metric="accuracy")
