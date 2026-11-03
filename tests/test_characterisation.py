"""Characterisation: outputs must equal those of the pre-refactor code.

tests/golden/baseline.json was produced by running tests/capture.py against the
ORIGINAL modules (main @ ed2f833) on deterministic synthetic panels. All
offline; no live data API is touched.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests.capture import capture_all

# Long decimals are rounded to 10 significant digits before comparing: the last 1-2 digits of
# DuckDB/pandas aggregates differ between library builds (local vs CI runner), which is
# floating-point noise, not a behaviour change.
_LONG_FLOAT = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def _norm(obj):
    if isinstance(obj, dict):
        return {k: _norm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_norm(v) for v in obj]
    if isinstance(obj, str):
        return _LONG_FLOAT.sub(lambda m: f"{float(m.group()):.10g}", obj)
    if isinstance(obj, float):
        return float(f"{obj:.10g}")
    return obj


GOLDEN = json.loads((Path(__file__).parent / "golden" / "baseline.json").read_text())


@pytest.fixture(scope="module")
def current():
    return json.loads(json.dumps(capture_all(), sort_keys=True))


@pytest.mark.parametrize("key", ["piotroski", "misc"])
def test_pure_functions_unchanged(current, key):
    assert _norm(current[key]) == _norm(GOLDEN[key])


def test_us_sweep_stdout_and_csv_unchanged(current):
    assert _norm(current["sweep_us"]) == _norm(GOLDEN["sweep_us"])


def test_india_sweep_stdout_and_csv_unchanged(current):
    assert _norm(current["sweep_in"]) == _norm(GOLDEN["sweep_in"])


def test_daily_breakout_stdout_and_csv_unchanged():
    from tests import synth
    from tests.capture import run_sweep

    golden = json.loads((Path(__file__).parent / "golden" / "daily_breakout.json").read_text())
    now = json.loads(json.dumps(run_sweep("daily_breakout_combo_us", synth.us_panels), sort_keys=True))
    assert _norm(now) == _norm(golden)
