from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(autouse=True)
def _isolate_runtime_state(tmp_path_factory, monkeypatch):
    """Plasă de siguranță: niciun test nu scrie în starea reală a proiectului.

    Înainte, suita adăuga intrări „test" în ledger-ul de decizii real și îngheța
    un snapshot fals (metric 1.0) în forward_test.json al Multi Multi — pe care
    Reality Check îl afișa apoi ca rezultat real.
    """
    import decision_ledger
    import forward_test

    sandbox = tmp_path_factory.mktemp("runtime_state")
    monkeypatch.setattr(decision_ledger, "DATA_DIR", sandbox)
    monkeypatch.setattr(forward_test, "snapshot_path",
                        lambda lottery_id: sandbox / lottery_id / forward_test.SNAPSHOT_FILE)
