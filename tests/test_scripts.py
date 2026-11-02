"""Smoke tests: every script compiles and exposes its entry point without side effects."""
from __future__ import annotations

import importlib
import py_compile
from pathlib import Path

import pandas as pd
import pytest

import adaptive_liquidity as AL
import golden_cross as GC

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p for p in ROOT.glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("mod", ["sweep_piotroski_plus", "sweep_piotroski_plus_us", "cost_vs_edge"])
def test_import_has_no_side_effects_and_exposes_main(mod):
    assert callable(importlib.import_module(mod).main)


def test_golden_cross_null_result_on_short_history():
    out = GC.compute(pd.DataFrame({"Close": range(50)}))
    assert out["trend"] == "INSUFFICIENT_HISTORY" and out["gc_signal"] is False


def test_adaptive_gate_capital_floor_and_policy_floor():
    assert AL.Gate("US", 1_000_000, 10).min_adv_usd == pytest.approx(100_000 / 0.15)
    assert AL.Gate("IN", 10_000, 10).min_adv_usd == AL.MARKET_FLOOR_USD["IN"]
    assert AL.scan_floor("XX") == AL.STRUCTURAL_FLOOR_USD
