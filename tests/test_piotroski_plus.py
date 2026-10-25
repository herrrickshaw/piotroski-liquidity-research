"""Tests for piotroski_plus.score / weigh with a fake yfinance Ticker (no network)."""
import math

import numpy as np
import pandas as pd
import pytest

import piotroski_plus as PP

Y = [pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-31"), pd.Timestamp("2022-12-31")]


class FakeTicker:
    def __init__(self, inc, bal, cfs):
        self.income_stmt = pd.DataFrame(inc, index=Y).T
        self.balance_sheet = pd.DataFrame(bal, index=Y).T
        self.cashflow = pd.DataFrame(cfs, index=Y).T


def _good():
    # every Piotroski test passes in 2024 vs 2023
    inc = {"EBIT": [30, 20, 18], "Net Income": [20, 10, 9],
           "Total Revenue": [200, 150, 140], "Gross Profit": [100, 60, 56]}
    bal = {"Total Assets": [200, 200, 190], "Current Liabilities": [50, 60, 60],
           "Current Assets": [100, 90, 85], "Long Term Debt": [20, 40, 40],
           "Ordinary Shares Number": [100, 100, 100],
           "Cash And Cash Equivalents": [10, 10, 10]}
    cfs = {"Operating Cash Flow": [30, 15, 12]}
    return FakeTicker(inc, bal, cfs)


def test_all_nine_pass_hand_computed():
    r = PP.score(_good())
    assert r["f_score"] == 9 and r["f_tested"] == 9
    for n in PP.PIOTROSKI_TESTS:
        assert r[n] is True, n


def test_roce_block_hand_computed():
    r = PP.score(_good())
    hist = [30 / 150, 20 / 140, 18 / 130]
    assert r["roce"] == pytest.approx(30 / 150)
    assert r["roce_ex_cash"] == pytest.approx(30 / 140)     # cash removed from CE
    assert r["roce_cv"] == pytest.approx(np.std(hist) / np.mean(hist))
    assert r["10_roce_level"] is True                        # 21.4% > 15%
    assert r["11_roce_stable"] is True                       # cv ~0.15 < 0.30
    assert r["12_roce_not_deteriorating"] is True
    assert r["plus_score"] == 3 and r["total"] == 12


def test_missing_lines_are_skipped_not_failed():
    t = _good()
    t.income_stmt = t.income_stmt.drop(index="Gross Profit")
    t.balance_sheet = t.balance_sheet.drop(index="Long Term Debt")
    r = PP.score(t)
    assert r["8_gross_margin_rising"] is None and r["5_leverage_falling"] is None
    assert r["f_tested"] == 7 and r["f_score"] == 7


def test_dilution_tolerates_one_percent():
    t = _good()
    t.balance_sheet.loc["Ordinary Shares Number", Y[0]] = 101
    assert PP.score(t)["7_no_dilution"] is True
    t.balance_sheet.loc["Ordinary Shares Number", Y[0]] = 102
    assert PP.score(t)["7_no_dilution"] is False


def test_no_statements_returns_empty():
    t = _good()
    t.income_stmt = pd.DataFrame()
    assert PP.score(t) == {}


def test_weigh_canonical_normalises_by_tests_that_ran():
    r = {n: True for n in PP.TEST_NAMES}
    r["8_gross_margin_rising"] = None
    r["1_roa_positive"] = False
    w = PP.weigh(r, "canonical")
    assert w["possible"] == 11 and w["raw"] == 10
    assert w["pct"] == pytest.approx(10 / 11 * 100)
    assert w["coverage"] == pytest.approx(11 / 12)


def test_weigh_round_tripped_nan_and_strings():
    r = {n: "True" for n in PP.TEST_NAMES}
    r["2_cfo_positive"] = float("nan")       # skipped test after CSV round trip
    r["3_roa_improving"] = "False"
    w = PP.weigh(r, "canonical")
    assert w["possible"] == 11 and w["raw"] == 10


def test_weigh_custom_weights_and_errors():
    r = {"1_roa_positive": True, "2_cfo_positive": False}
    w = PP.weigh(r, {"1_roa_positive": 3.0, "2_cfo_positive": 1.0})
    assert w["pct"] == pytest.approx(75.0)
    with pytest.raises(KeyError):
        PP.weigh(r, "nope")
    assert PP.weigh({}, "canonical")["pct"] is None
    assert PP.weigh({"x": True}, "canonical")["pct"] is None


def test_presets_cover_only_known_tests():
    for name, vec in PP.PRESETS.items():
        assert set(vec) <= set(PP.TEST_NAMES), name
        assert all(w >= 0 for w in vec.values())
