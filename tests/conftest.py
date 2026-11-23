"""Test plumbing: put the repo root (flat scripts) on sys.path and import scripts safely."""
import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


@pytest.fixture
def load(monkeypatch):
    """Import a repo script fresh, with a chosen sys.argv, and forget it afterwards.

    Several scripts read sys.argv at import time (PER_TIER, N); pytest's own argv
    would break them, so argv is pinned. sys.modules is restored on teardown so a
    module patched by one test never leaks into the next.
    """
    before = set(sys.modules)

    def _load(name, *argv):
        monkeypatch.setattr(sys, "argv", [name, *argv])
        sys.modules.pop(name, None)
        return importlib.import_module(name)

    yield _load
    for k in set(sys.modules) - before:
        sys.modules.pop(k, None)
    sys.modules.pop("piotroski_plus", None)
