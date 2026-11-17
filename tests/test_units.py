"""Unit tests for the pure helpers touched by the refactor. Offline, deterministic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import piotroski_plus as PP
import sweep_common as SC
from tests import synth


def test_ratio_gt_lt_skip_on_missing_or_zero_denominator():
    assert PP._ratio(1.0, 4.0) == 0.25
    assert PP._ratio(None, 4.0) is None and PP._ratio(1.0, 0) is None
    assert PP._gt(2, 1) is True and PP._gt(None, 1) is None
    assert PP._lt(1, 2) is True and PP._lt(1, None) is None


def test_tally_counts_only_tests_that_ran():
    detail: dict = {}
    t = PP._Tally(detail)
    t("a", True)
    t("b", False)
    t("c", None)
    assert (t.passed, t.ran) == (1, 2)
    assert detail == {"a": True, "b": False, "c": None}


def test_score_survives_statement_fetch_failure():
    class Boom:
        @property
        def income_stmt(self):
            raise RuntimeError("network down")

    assert PP.score(Boom()) == {}


def test_score_skips_missing_line_instead_of_failing_it():
    full = PP.score(synth.fake_ticker(1))
    no_gp = PP.score(synth.fake_ticker(1, drop=("Gross Profit",)))
    assert full["f_tested"] == 9 and no_gp["f_tested"] == 8
    assert no_gp["8_gross_margin_rising"] is None


def test_weigh_normalises_by_tests_that_ran_and_handles_csv_nan():
    r = {"a": "True", "b": float("nan"), "c": "False"}
    w = PP.weigh(r, {"a": 2.0, "b": 5.0, "c": 1.0})
    assert w["raw"] == 2.0 and w["possible"] == 3.0
    assert w["pct"] == pytest.approx(200 / 3) and w["coverage"] == pytest.approx(3 / 8)


def test_weigh_unknown_preset_raises():
    with pytest.raises(KeyError):
        PP.weigh({"a": True}, "nope")


def test_verdict_labels():
    base = {"f_tested": 9, "plus_tested": 3}
    assert PP.verdict({**base, "f_score": 8, "plus_score": 3}) == "STRONG"
    assert PP.verdict({**base, "f_score": 8, "plus_score": 0}) == "IMPROVING_WEAK_BUSINESS"
    assert PP.verdict({**base, "f_score": 2, "plus_score": 1}) == "AVOID"
    assert PP.verdict({}) == "INSUFFICIENT_DATA"


def test_roce_stability_tests():
    assert SC.roce_stability_tests(0.2, [0.2, 0.2]) == (None, None)       # <3 years
    assert SC.roce_stability_tests(0.2, [0.0, 0.0, 0.0]) == (None, None)  # ~0 mean
    assert SC.roce_stability_tests(0.21, [0.2, 0.21, 0.19]) == (True, True)
    assert SC.roce_stability_tests(0.05, [0.5, 0.05, 0.3]) == (False, False)


def test_forward_returns_flags_splits_and_anchors_to_next_trading_day():
    import duckdb
    dates = pd.bdate_range("2020-01-01", periods=30)
    px = pd.concat([
        pd.DataFrame({"Date": dates, "Symbol": "OK", "Close": np.linspace(100, 130, 30)}),
        pd.DataFrame({"Date": dates, "Symbol": "SPLIT", "Close": [100.0] * 15 + [1.0] * 15}),
    ])
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/px.parquet"
        px.to_parquet(path)
        sunday = pd.Timestamp("2020-01-05")          # falls on a weekend
        ret = SC.forward_returns(duckdb.connect(), path, [sunday], hold=20)
    assert set(ret.Symbol) == {"OK"}                  # split window excluded
    ok = px[px.Symbol == "OK"].Close
    assert ret.fwd_ret.iloc[0] == pytest.approx(ok.iloc[23] / ok.iloc[3] - 1)
