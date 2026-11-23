import math

import numpy as np
import pytest
from helpers import FakeTicker, YEARS, perfect_ticker, stmt

import piotroski_plus as PP

ALL = PP.TEST_NAMES


def over(**kw):
    return PP.score(perfect_ticker(**kw))


def test_perfect_company_passes_every_test():
    r = PP.score(perfect_ticker())
    assert (r["f_score"], r["f_tested"]) == (9, 9)
    assert (r["plus_score"], r["plus_tested"]) == (3, 3)
    assert r["total"] == 12
    assert all(r[n] is True for n in ALL)
    assert r["years"] == 4
    # ROCE = EBIT / (TA - CL) = 200 / 800 ; ex-cash removes the 100 of cash
    assert r["roce"] == pytest.approx(200 / 800)
    assert r["roce_ex_cash"] == pytest.approx(200 / 700)
    hist = [200 / 800, 150 / 800, 120 / 800, 100 / 800]
    assert r["roce_cv"] == pytest.approx(np.std(hist) / np.mean(hist))


@pytest.mark.parametrize("kw,failing", [
    # profitability
    (dict(Net__Income=[-10, 60, 50, 40]), {"1_roa_positive", "3_roa_improving"}),
    (dict(Operating__Cash__Flow=[-5, 90, 80, 70]), {"2_cfo_positive", "4_accruals_cfo_gt_roa"}),
    (dict(Net__Income=[50, 60, 50, 40]), {"3_roa_improving"}),
    (dict(Operating__Cash__Flow=[50, 90, 80, 70]), {"4_accruals_cfo_gt_roa"}),
    # leverage / liquidity / dilution
    (dict(Long__Term__Debt=[300, 200, 200, 200]), {"5_leverage_falling"}),
    (dict(Current__Assets=[300, 400, 400, 400]), {"6_current_ratio_rising"}),
    (dict(Ordinary__Shares__Number=[102, 100, 100, 100]), {"7_no_dilution"}),
    # efficiency
    (dict(Gross__Profit=[300, 300, 250, 200]), {"8_gross_margin_rising"}),
    (dict(Total__Revenue=[850, 900, 800, 700]), {"9_asset_turnover_rising"}),
])
def test_piotroski_tests_fail_exactly_where_expected(kw, failing):
    r = over(**kw)
    assert {n for n in PP.PIOTROSKI_TESTS if r[n] is False} == failing
    assert r["f_score"] == 9 - len(failing)
    assert r["f_tested"] == 9 and r["plus_score"] == 3


def test_dilution_tolerance_is_one_percent_inclusive():
    assert over(Ordinary__Shares__Number=[101, 100, 100, 100])["7_no_dilution"] is True
    assert over(Ordinary__Shares__Number=[102, 100, 100, 100])["7_no_dilution"] is False


def test_roce_level_hurdle_is_strict_and_uses_ex_cash():
    # ce = 800, cash zero: EBIT 120 -> exactly the 15% hurdle -> NOT a pass; 121 passes
    base = dict(Cash__And__Cash__Equivalents=[0, 0, 0, 0])
    assert over(EBIT=[120, 150, 120, 100], **base)["10_roce_level"] is False
    assert over(EBIT=[121, 150, 120, 100], **base)["10_roce_level"] is True
    # cash is NOT capital employed: same EBIT, big cash balance lifts roce_ex_cash above roce
    r = over(EBIT=[100, 150, 120, 100], Cash__And__Cash__Equivalents=[400, 0, 0, 0])
    assert r["roce"] == pytest.approx(100 / 800)
    assert r["roce_ex_cash"] == pytest.approx(100 / 400)
    assert r["10_roce_level"] is True


def test_roce_ex_cash_is_none_when_cash_exceeds_capital():
    r = over(Cash__And__Cash__Equivalents=[900, 0, 0, 0])
    assert r["roce_ex_cash"] is None
    assert r["10_roce_level"] is None
    assert r["plus_tested"] == 2


def test_cash_line_missing_falls_back_to_zero():
    r = over(Cash__And__Cash__Equivalents=None)
    assert r["roce_ex_cash"] == pytest.approx(r["roce"])


def test_roce_stability_uses_cv_hurdle():
    steady = over(EBIT=[200, 200, 200, 200])
    assert steady["roce_cv"] == pytest.approx(0.0)
    assert steady["11_roce_stable"] is True
    erratic = over(EBIT=[200, 40, 240, 20])
    assert erratic["roce_cv"] > PP.ROCE_CV_HURDLE
    assert erratic["11_roce_stable"] is False


def test_roce_not_deteriorating_is_inclusive_of_the_mean():
    # every year equal -> latest == mean -> still passes (>=)
    assert over(EBIT=[200, 200, 200, 200])["12_roce_not_deteriorating"] is True
    assert over(EBIT=[100, 200, 200, 200])["12_roce_not_deteriorating"] is False


def test_history_needs_three_years_for_stability_and_trend():
    two = PP.score(perfect_ticker_two_years())
    assert two["11_roce_stable"] is None and two["12_roce_not_deteriorating"] is None
    assert two["roce_cv"] is None
    assert (two["plus_score"], two["plus_tested"]) == (1, 1)
    assert two["total"] == two["f_score"] + 1


def perfect_ticker_two_years():
    t = perfect_ticker()
    cut = lambda df: df.iloc[:, :2]
    return FakeTicker(cut(t.income_stmt), cut(t.balance_sheet), cut(t.cashflow))


def test_three_years_is_enough():
    t = perfect_ticker()
    cut = lambda df: df.iloc[:, :3]
    r = PP.score(FakeTicker(cut(t.income_stmt), cut(t.balance_sheet), cut(t.cashflow)))
    assert r["years"] == 3
    assert r["11_roce_stable"] is True


def test_near_zero_mean_roce_gives_no_cv():
    r = over(EBIT=[4, 4, 4, 4])     # roce 0.005 -> |mean| <= 0.01
    assert r["roce_cv"] is None
    assert r["11_roce_stable"] is None
    assert r["12_roce_not_deteriorating"] is True    # trend test still runs


def test_year_with_nonpositive_capital_is_excluded_from_history():
    r = over(Current__Liabilities=[200, 200, 200, 1200])
    assert r["years"] == 3
    r0 = over(Current__Liabilities=[1200, 200, 200, 200])
    assert r0["roce"] is None
    assert r0["10_roce_level"] is None and r0["12_roce_not_deteriorating"] is None


def test_missing_inputs_are_skipped_not_scored_zero():
    r = over(Gross__Profit=None)
    assert r["8_gross_margin_rising"] is None
    assert (r["f_score"], r["f_tested"]) == (8, 8)
    r = over(Long__Term__Debt=None, Ordinary__Shares__Number=None)
    assert (r["f_score"], r["f_tested"]) == (7, 7)
    assert r["5_leverage_falling"] is None and r["7_no_dilution"] is None


def test_zero_total_assets_skips_ratio_tests_without_crashing():
    r = over(Total__Assets=[0, 1000, 1000, 1000])
    for n in ("1_roa_positive", "3_roa_improving", "4_accruals_cfo_gt_roa",
              "5_leverage_falling", "9_asset_turnover_rising"):
        assert r[n] is None


def test_alternate_line_item_names_are_accepted():
    t = perfect_ticker()
    ren = lambda df, m: df.rename(index=m)
    inc = ren(t.income_stmt, {"EBIT": "Operating Income", "Net Income": "NetIncome",
                              "Total Revenue": "Operating Revenue"})
    bal = ren(t.balance_sheet, {"Current Liabilities": "Total Current Liabilities",
                                "Current Assets": "Total Current Assets",
                                "Ordinary Shares Number": "Share Issued"})
    cfs = ren(t.cashflow, {"Operating Cash Flow": "Total Cash From Operating Activities"})
    r = PP.score(FakeTicker(inc, bal, cfs))
    assert (r["f_score"], r["plus_score"]) == (9, 3)


@pytest.mark.parametrize("drop", ["EBIT", "Total Assets", "Current Liabilities"])
def test_core_lines_missing_returns_empty(drop):
    assert PP.score(perfect_ticker(**{drop.replace(" ", "__"): None})) == {}


def test_single_overlapping_year_returns_empty():
    t = perfect_ticker()
    one = lambda df: df.iloc[:, :1]
    assert PP.score(FakeTicker(one(t.income_stmt), one(t.balance_sheet), one(t.cashflow))) == {}


def test_statement_fetch_error_returns_empty():
    assert PP.score(FakeTicker(boom=True)) == {}


def test_empty_or_none_frames_return_empty():
    assert PP.score(FakeTicker(None, None, None)) == {}
    import pandas as pd
    assert PP.score(FakeTicker(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())) == {}


def test_overlap_uses_years_present_in_all_three_statements():
    t = perfect_ticker()
    bal = t.balance_sheet.drop(columns=t.balance_sheet.columns[0])   # newest year missing from balance
    r = PP.score(FakeTicker(t.income_stmt, bal, t.cashflow))
    assert r["years"] == 3


# ── weigh / sweep / _ran / _passed ───────────────────────────────────────────
def test_presets_cover_all_tests_and_canonical_is_all_ones():
    assert PP.PIOTROSKI_TESTS == ALL[:9] and PP.ROCE_TESTS == ALL[9:]
    assert len(ALL) == 12 and PP.F_MAX == 9 and PP.PLUS_MAX == 3
    assert set(PP.PRESETS) == {"canonical", "quality", "turnaround", "safety"}
    assert PP.PRESETS["canonical"] == {n: 1.0 for n in ALL}
    for w in PP.PRESETS.values():
        assert set(w) == set(ALL)


def test_weigh_perfect_and_failing_company():
    r = PP.score(perfect_ticker())
    w = PP.weigh(r)
    assert w == {"raw": 12.0, "possible": 12.0, "pct": 100.0, "coverage": 1.0}
    bad = dict(r, **{"1_roa_positive": False, "10_roce_level": False})
    w = PP.weigh(bad, "canonical")
    assert (w["raw"], w["possible"]) == (10.0, 12.0)
    assert w["pct"] == pytest.approx(10 / 12 * 100)


def test_weigh_weights_decide_what_a_pass_is_worth():
    r = {n: True for n in ALL}
    r["11_roce_stable"] = False
    q = PP.PRESETS["quality"]
    w = PP.weigh(r, "quality")
    assert w["raw"] == pytest.approx(sum(q.values()) - q["11_roce_stable"])
    assert w["possible"] == pytest.approx(sum(q.values()))
    assert w["pct"] == pytest.approx(w["raw"] / w["possible"] * 100)


def test_weigh_skipped_tests_leave_numerator_and_denominator():
    r = {n: True for n in ALL}
    r["8_gross_margin_rising"] = None
    r["6_current_ratio_rising"] = float("nan")
    w = PP.weigh(r, "canonical")
    assert (w["raw"], w["possible"]) == (10.0, 10.0)
    assert w["pct"] == 100.0
    assert w["coverage"] == pytest.approx(10 / 12)


def test_weigh_dict_weights_and_unlisted_tests_weigh_zero():
    r = {"1_roa_positive": True, "2_cfo_positive": False}
    w = PP.weigh(r, {"1_roa_positive": 3.0, "2_cfo_positive": 1.0})
    assert (w["raw"], w["possible"], w["pct"], w["coverage"]) == (3.0, 4.0, 75.0, 1.0)
    # a test absent from the vector contributes nothing, even if it failed
    r["3_roa_improving"] = False
    assert PP.weigh(r, {"1_roa_positive": 3.0, "2_cfo_positive": 1.0})["pct"] == 75.0


def test_weigh_unknown_preset_and_empty_input():
    with pytest.raises(KeyError, match="unknown preset"):
        PP.weigh({"1_roa_positive": True}, "nope")
    assert PP.weigh({}) == {"raw": None, "possible": None, "pct": None, "coverage": None}
    assert PP.weigh(None) == {"raw": None, "possible": None, "pct": None, "coverage": None}


def test_weigh_nothing_ran_gives_zero_possible_and_no_pct():
    w = PP.weigh({n: None for n in ALL})
    assert w == {"raw": 0.0, "possible": 0.0, "pct": None, "coverage": 0.0}
    # weights summing to zero must not divide by zero
    w = PP.weigh({"1_roa_positive": True}, {"1_roa_positive": 0.0})
    assert w["pct"] is None and w["possible"] == 0.0


def test_csv_round_trip_values_do_not_count_skipped_tests_as_passes():
    # NaN is truthy, "False" is truthy: both were real bugs before _ran/_passed existed
    r = {"1_roa_positive": "True", "2_cfo_positive": "False", "3_roa_improving": float("nan"),
         "4_accruals_cfo_gt_roa": "1", "5_leverage_falling": "0", "6_current_ratio_rising": " TRUE "}
    w = PP.weigh(r, "canonical")
    assert (w["raw"], w["possible"]) == (3.0, 5.0)


@pytest.mark.parametrize("v,ran", [
    (None, False), (float("nan"), False), (True, True), (False, True), (0, True), (0.5, True),
    ("true", True), ("False", True), ("1", True), ("0", True), (" TRUE ", True),
    ("maybe", False), ("", False), ("nan", False)])
def test_ran(v, ran):
    assert PP._ran(v) is ran


@pytest.mark.parametrize("v,passed", [
    (True, True), (False, False), (1, True), (0, False), ("True", True), ("true", True),
    ("1", True), (" 1 ", True), ("False", False), ("0", False), ("", False), ("yes", False)])
def test_passed(v, passed):
    assert PP._passed(v) is passed


def test_sweep_returns_every_preset_or_the_requested_subset():
    r = PP.score(perfect_ticker())
    s = PP.sweep(r)
    assert set(s) == set(PP.PRESETS) and all(v == 100.0 for v in s.values())
    assert set(PP.sweep(r, ["safety"])) == {"safety"}
    assert PP.sweep({}, ["quality"]) == {"quality": None}


def test_quality_preset_prefers_stability_over_level():
    # steady-but-regulated (stable only) vs erratic-but-high (level only) tie under equal weights
    steady = {n: False for n in ALL}; steady["11_roce_stable"] = True
    high = {n: False for n in ALL}; high["10_roce_level"] = True
    assert PP.weigh(steady)["pct"] == PP.weigh(high)["pct"]
    assert PP.weigh(steady, "quality")["pct"] > PP.weigh(high, "quality")["pct"]


# ── verdict ──────────────────────────────────────────────────────────────────
def V(f, ft, p, pt):
    return PP.verdict({"f_score": f, "f_tested": ft, "plus_score": p, "plus_tested": pt})


def test_verdict_insufficient_data_rules():
    assert PP.verdict({}) == "INSUFFICIENT_DATA"
    assert PP.verdict(None) == "INSUFFICIENT_DATA"
    assert PP.verdict({"f_score": None, "plus_score": 3, "f_tested": 9, "plus_tested": 3}) == "INSUFFICIENT_DATA"
    assert PP.verdict({"f_score": 9, "plus_score": None, "f_tested": 9, "plus_tested": 3}) == "INSUFFICIENT_DATA"
    assert V(9, 5, 3, 3) == "INSUFFICIENT_DATA"        # f_tested < 6
    assert V(6, 6, 3, 3) == "STRONG"                   # 6 tests is enough (9-point scaled)
    assert V(9, 9, 3, 1) == "INSUFFICIENT_DATA"        # plus_tested < 2
    assert V(9, 9, 3, 2) == "STRONG"
    assert V(0, 9, 0, 2) == "AVOID"                    # zero score is data, not missing data


def test_verdict_labels_and_thresholds():
    assert V(7, 9, 3, 3) == "STRONG"
    assert V(7, 9, 2, 3) == "GOOD"
    assert V(7, 9, 1, 3) == "IMPROVING_WEAK_BUSINESS"
    assert V(7, 9, 0, 3) == "IMPROVING_WEAK_BUSINESS"
    assert V(6, 9, 3, 3) == "QUALITY_NOT_IMPROVING"   # fs 6 < 7, ROCE perfect
    assert V(4, 9, 3, 3) == "QUALITY_NOT_IMPROVING"   # fs >= 4 inclusive
    assert V(3, 9, 3, 3) == "MIXED"                   # fs 3 < 4
    assert V(3, 9, 1, 3) == "AVOID"                   # fs <= 3 inclusive, p <= 1
    assert V(3, 9, 0, 3) == "AVOID"
    assert V(3, 9, 2, 3) == "MIXED"                   # p = 2 escapes AVOID
    assert V(4, 9, 1, 3) == "MIXED"
    assert V(6, 9, 2, 3) == "MIXED"


def test_verdict_scales_to_a_nine_point_frame_when_tests_were_skipped():
    assert V(6, 7, 3, 3) == "STRONG"        # 6/7*9 = 7.71
    assert V(5, 7, 3, 3) == "QUALITY_NOT_IMPROVING"   # 6.43 < 7
    assert V(2, 6, 1, 3) == "AVOID"         # 2/6*9 = 3.0 -> <= 3
    assert V(3, 8, 3, 3) == "MIXED"         # 3/8*9 = 3.375 > 3 and < 4 -> MIXED
    assert V(4, 8, 3, 3) == "QUALITY_NOT_IMPROVING"   # 4.5 >= 4
