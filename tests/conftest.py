import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("ISRM_SANDBOX_QUICK", "1")


@pytest.fixture
def tmp_ledger(tmp_path):
    return tmp_path / "ledger.jsonl"
