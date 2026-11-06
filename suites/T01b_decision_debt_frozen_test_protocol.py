"""T01b: the decision-debt test under the original simulation's protocol (debt frozen
during the test). Same world and controllers as T01; the spec's controller
configuration carries freeze_debt_in_test = True."""
from __future__ import annotations

from T01_decision_debt_history_dependence import run  # noqa: F401
