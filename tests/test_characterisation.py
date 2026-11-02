"""Characterisation: outputs must equal those of the pre-refactor code.

tests/golden/baseline.json was produced by running tests/capture.py against the
ORIGINAL modules (main @ ed2f833) on deterministic synthetic panels. All
offline; no live data API is touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.capture import capture_all

GOLDEN = json.loads((Path(__file__).parent / "golden" / "baseline.json").read_text())


@pytest.fixture(scope="module")
def current():
    return json.loads(json.dumps(capture_all(), sort_keys=True))


@pytest.mark.parametrize("key", ["piotroski", "misc"])
def test_pure_functions_unchanged(current, key):
    assert current[key] == GOLDEN[key]


def test_us_sweep_stdout_and_csv_unchanged(current):
    assert current["sweep_us"] == GOLDEN["sweep_us"]


def test_india_sweep_stdout_and_csv_unchanged(current):
    assert current["sweep_in"] == GOLDEN["sweep_in"]
