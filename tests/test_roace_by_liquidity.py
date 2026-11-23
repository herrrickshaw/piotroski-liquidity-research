import datetime as dt

import numpy as np
import pandas as pd
import pytest
from helpers import PANDAS3, FakeTicker, YEARS, bdays, flat_ticker, install_fake_yfinance, stmt, write_parquet

END = pd.Timestamp("2026-06-30")

# roace_by_liquidity.py:247-249 builds the sample with groupby("tier").apply(...). On pandas>=3 the
# grouping column is no longer passed to the applied function, so 'tier' is missing from the sample
# and main() dies with KeyError at the first keep["tier"] lookup.
pandas3_tier_bug = pytest.mark.xfail(
    PANDAS3, strict=True, raises=KeyError,
    reason="roace_by_liquidity.py:247 groupby('tier').apply drops the grouping column on pandas>=3")


@pytest.fixture
def rbl(load, monkeypatch, tmp_path):
    mod = load("roace_by_liquidity", "5")
    (tmp_path / "reports").mkdir()
    monkeypatch.chdir(tmp_path)
    return mod


def series(sym, vol, n=80, end=END, close=100.0, valid=None):
    d = pd.bdate_range(end=end, periods=n)
    df = pd.DataFrame({"Date": d, "Symbol": sym, "Close": close, "Volume": float(vol)})
    return df if valid is None else df.iloc[-valid:]


def universe_frames(per_tier=3):
    frames = []
    for ti, base in enumerate((300, 20, 1)):                    # LARGE, MID, SMALL volume scales
        for k in range(per_tier):
            frames.append(series(f"{'LMS'[ti]}{k}", base * 1000 * (k + 1)))
    return frames


# ── tiers() ──────────────────────────────────────────────────────────────────
def test_tiers_split_into_equal_thirds_by_turnover(rbl, tmp_path):
    rbl.PARQUET = write_parquet(pd.concat(universe_frames(), ignore_index=True), tmp_path / "in.parquet")
    t = rbl.tiers().set_index("Symbol")
    assert len(t) == 9
    assert set(t[t.tier == "LARGE"].index) == {"L0", "L1", "L2"}
    assert set(t[t.tier == "MID"].index) == {"M0", "M1", "M2"}
    assert set(t[t.tier == "SMALL"].index) == {"S0", "S1", "S2"}
    assert t.turnover["L2"] == pytest.approx(100.0 * 300 * 1000 * 3)      # median of Close*Volume


def test_tiers_filters_thin_dead_and_invalid_rows(rbl, tmp_path):
    frames = universe_frames()
    frames += [
        series("KEEP50", 5000, n=50),                                   # exactly 50 valid rows: kept
        series("THIN49", 5000, n=49),                                   # 49: dropped
        series("ALIVE", 5000, end=pd.Timestamp("2026-06-01")),         # last bar on the cutoff: kept
        series("DEAD", 5000, end=pd.Timestamp("2026-05-29")),          # stopped trading: dropped
        series("ZEROPX", 5000, close=0.0),
        series("ZEROVOL", 0),
    ]
    rbl.PARQUET = write_parquet(pd.concat(frames, ignore_index=True), tmp_path / "in.parquet")
    names = set(rbl.tiers().Symbol)
    assert {"KEEP50", "ALIVE"} <= names
    assert not {"THIN49", "DEAD", "ZEROPX", "ZEROVOL"} & names


def test_tiers_turnover_uses_only_the_latest_sixty_valid_bars(rbl, tmp_path):
    old = series("X", 1.0, n=80)
    old.loc[old.index[:20], "Volume"] = 1e9                              # huge volume, but > 60 bars ago
    rbl.PARQUET = write_parquet(pd.concat([old] + universe_frames()[:2], ignore_index=True), tmp_path / "in.parquet")
    t = rbl.tiers().set_index("Symbol")
    assert t.turnover["X"] == pytest.approx(100.0)


# ── _series ──────────────────────────────────────────────────────────────────
def test_series_helper(load):
    f = load("roace_by_liquidity")._series
    df = stmt({"A": [1.0, None, 3.0], "B": [7, 8, 9]}, ["2025-03-31", "2024-03-31", "2023-03-31"])
    r = f(df, "Z", "A")
    assert r.tolist() == [1.0, 3.0] and r.dtype == float            # NaN dropped, first matching name wins
    assert f(df, "Z") is None
    assert f(None, "A") is None
    assert f(pd.DataFrame(), "A") is None
    assert f(stmt({"A": [None, None]}, ["2025-03-31", "2024-03-31"]), "A") is None


# ── roace_one ────────────────────────────────────────────────────────────────
def one(rbl, monkeypatch, ticker, sym="X"):
    install_fake_yfinance(monkeypatch, {f"{sym}.NS": ticker})
    return rbl.roace_one(sym)


def test_roace_levels_and_cash_adjustment(rbl, monkeypatch):
    r = one(rbl, monkeypatch, flat_ticker(ebit=160, cash=400))
    assert r["Symbol"] == "X" and r["years"] == 4
    assert r["ebit"] == 160 and r["capital_employed"] == 800 and r["cash"] == 400
    assert r["roace"] == pytest.approx(0.2)                       # 160 / (1000-200)
    assert r["roace_ex_cash"] == pytest.approx(0.4)               # 160 / (800-400)
    assert r["cash_pct_of_ce"] == pytest.approx(50.0)
    assert r["roace_cv"] == pytest.approx(0.0)


def test_roace_ex_cash_none_when_cash_swallows_capital(rbl, monkeypatch):
    r = one(rbl, monkeypatch, flat_ticker(cash=800.0))
    assert r["roace_ex_cash"] is None
    r = one(rbl, monkeypatch, flat_ticker(cash=900.0))
    assert r["roace_ex_cash"] is None


def test_cash_line_missing_means_zero_cash(rbl, monkeypatch):
    r = one(rbl, monkeypatch, flat_ticker(Cash__And__Cash__Equivalents=None))
    assert r["cash"] == 0.0 and r["roace_ex_cash"] == pytest.approx(r["roace"])


def test_roace_cv_needs_three_years_and_a_nonzero_mean(rbl, monkeypatch):
    hist = [160 / 800, 100 / 800, 40 / 800]
    r = one(rbl, monkeypatch, flat_ticker(ebits=[160, 100, 40]))
    assert r["years"] == 3 and r["roace_cv"] == pytest.approx(np.std(hist) / np.mean(hist))
    assert one(rbl, monkeypatch, flat_ticker(ebits=[160, 100]))["roace_cv"] is None
    assert one(rbl, monkeypatch, flat_ticker(ebits=[4, 4, 4]))["roace_cv"] is None      # |mean| <= 0.01
    assert one(rbl, monkeypatch, flat_ticker(ebits=[10, 10, 10]))["roace_cv"] == pytest.approx(0.0)
    # a negative mean is measured against its magnitude
    neg = one(rbl, monkeypatch, flat_ticker(ebits=[-160, -100, -40]))
    assert neg["roace_cv"] == pytest.approx(np.std(hist) / abs(np.mean([-x for x in hist])))


def test_years_with_nonpositive_capital_are_left_out_of_history(rbl, monkeypatch):
    t = flat_ticker(ebits=[160, 100, 40, 20], Current__Liabilities=[200, 200, 200, 1500])
    r = one(rbl, monkeypatch, t)
    assert r["years"] == 3
    assert one(rbl, monkeypatch, flat_ticker(Current__Liabilities=[1500, 200, 200, 200])) is None
    assert one(rbl, monkeypatch, flat_ticker(Current__Liabilities=[1500] * 4)) is None


@pytest.mark.parametrize("drop", ["EBIT", "Total__Assets", "Current__Liabilities"])
def test_missing_core_statement_lines_give_none(rbl, monkeypatch, drop):
    assert one(rbl, monkeypatch, flat_ticker(**{drop: None})) is None


def test_ticker_errors_give_none(rbl, monkeypatch):
    assert one(rbl, monkeypatch, FakeTicker(boom=True)) is None
    install_fake_yfinance(monkeypatch, {})                           # Ticker(...) itself raises
    assert rbl.roace_one("MISSING") is None


def test_piotroski_seven_point_score_and_scaling(rbl, monkeypatch):
    t = flat_ticker(years=4)
    # constant statements: roa>0, cfo>0 pass; improving / leverage / current ratio / turnover do not
    r = one(rbl, monkeypatch, t)
    assert r["f_max"] == 7 and r["f_tested"] == 7
    assert r["f_score"] == pytest.approx(3.0)          # tests 1, 2 and 4 pass (cfo/ta .12 > roa .08)
    good = flat_ticker(Net__Income=[100.0, 60.0, 60.0, 60.0], Operating__Cash__Flow=[150.0, 120.0, 120.0, 120.0],
                       Long__Term__Debt=[50.0, 100.0, 100.0, 100.0],
                       Current__Assets=[600.0, 500.0, 500.0, 500.0],
                       Total__Revenue=[1200.0, 1000.0, 1000.0, 1000.0])
    r = one(rbl, monkeypatch, good)
    assert (r["f_score"], r["f_tested"]) == (7.0, 7)


def test_piotroski_skipped_tests_rescale_to_seven(rbl, monkeypatch):
    # drop long-term debt and current assets: 5 tests remain; score f*7/5
    t = flat_ticker(Long__Term__Debt=None, Current__Assets=None,
                    Net__Income=[100.0, 60.0, 60.0, 60.0], Operating__Cash__Flow=[150.0, 120.0, 120.0, 120.0],
                    Total__Revenue=[900.0, 1000.0, 1000.0, 1000.0])    # turnover falls -> test 7 fails
    r = one(rbl, monkeypatch, t)
    assert r["f_tested"] == 5
    assert r["f_score"] == pytest.approx(4 * 7 / 5)
    # four tests is too sparse to score at all
    t = flat_ticker(Long__Term__Debt=None, Current__Assets=None, Total__Revenue=None)
    r = one(rbl, monkeypatch, t)
    assert r["f_score"] is None and "f_tested" not in r and r["f_max"] == 7


def test_piotroski_needs_two_years_and_survives_missing_cashflow(rbl, monkeypatch):
    one_year = flat_ticker(years=1)
    r = one(rbl, monkeypatch, one_year)
    assert r["f_score"] is None and r["years"] == 1
    t = flat_ticker()

    class NoCash(FakeTicker):
        @property
        def cashflow(self):
            raise RuntimeError("boom")

    nc = NoCash(t.income_stmt, t.balance_sheet, None)
    r = one(rbl, monkeypatch, nc)
    assert r["f_tested"] == 5                    # tests 2 and 4 need cash flow


# ── main() ───────────────────────────────────────────────────────────────────
def build_main(rbl, monkeypatch, tmp_path, rocs, per_tier=3):
    """rocs: {symbol: Ticker}"""
    rbl.PARQUET = write_parquet(pd.concat(universe_frames(per_tier), ignore_index=True), tmp_path / "in.parquet")
    install_fake_yfinance(monkeypatch, {f"{s}.NS": t for s, t in rocs.items()})


def default_rocs(**special):
    # ROCE (EBIT/800): LARGE 10/20/30%, MID 12/24/36%, SMALL 5/15/25%
    table = {"L0": .10, "L1": .20, "L2": .30, "M0": .12, "M1": .24, "M2": .36, "S0": .05, "S1": .15, "S2": .25}
    out = {s: flat_ticker(ebit=r * 800) for s, r in table.items()}
    out.update(special)
    return out


@pandas3_tier_bug
def test_main_tables_by_tier(rbl, monkeypatch, tmp_path, capsys):
    build_main(rbl, monkeypatch, tmp_path, default_rocs())
    assert rbl.PER_TIER == 5
    assert rbl.main() == 0
    out = capsys.readouterr().out
    assert "ROACE by LIQUIDITY tier — India | sample 5/tier" in out
    assert "universe 9 live symbols -> sampling 9 (5/tier), 6 workers" in out
    assert "statements retrieved for 9/9 (100%)" in out
    sec1 = out.split("1. ROCE LEVEL")[1].split("2. ROCE STABILITY")[0]
    rows = {l.split()[0]: l.split() for l in sec1.splitlines() if l.split() and l.split()[0] in ("LARGE", "MID", "SMALL")}
    assert rows["LARGE"][1] == "3" and rows["LARGE"][3] == "20.0%" and rows["LARGE"][4] == "20.0%"
    assert rows["MID"][3] == "24.0%" and rows["SMALL"][3] == "15.0%"
    assert rows["LARGE"][5] == "0.0%"                                           # no cash -> cash%CE 0
    assert rows["LARGE"][6] == "15.0%" and rows["LARGE"][7] == "25.0%"           # p25 / p75 of 10,20,30
    # turnover Rs cr: median of Close*Volume / 1e7 -> LARGE: 100*300e3*2/1e7
    assert float(rows["LARGE"][2].replace(",", "")) == pytest.approx(100 * 300e3 * 2 / 1e7, abs=0.05)
    assert "dropped as out-of-range (banks/NBFCs/shells): 0" in out
    csv = pd.read_csv(tmp_path / "reports" / "roace_by_liquidity_india.csv")
    assert len(csv) == 9 and {"Symbol", "tier", "turnover", "roace_pct", "roace_ex_cash_pct"} <= set(csv.columns)
    assert csv.set_index("Symbol").roace_pct["M2"] == pytest.approx(36.0)
    assert "full sample -> reports/roace_by_liquidity_india.csv" in out


@pandas3_tier_bug
def test_main_stability_and_out_of_range_filter(rbl, monkeypatch, tmp_path, capsys):
    rocs = default_rocs(
        L2=flat_ticker(ebit=0.20 * 800 * 8),              # ROCE 160% -> outside [-50, 150]
        M1=flat_ticker(ebits=[200, 60, 190, 20]),       # erratic -> CV > 0.3
    )
    build_main(rbl, monkeypatch, tmp_path, rocs)
    assert rbl.main() == 0
    out = capsys.readouterr().out
    assert "dropped as out-of-range (banks/NBFCs/shells): 1" in out
    sec1 = out.split("1. ROCE LEVEL")[1].split("2. ROCE STABILITY")[0]
    large = [l.split() for l in sec1.splitlines() if l.strip().startswith("LARGE")][0]
    assert large[1] == "2"                                 # L2 dropped from the tier cut
    sec2 = out.split("2. ROCE STABILITY")[1].split("3. ROCE")[0]
    rows = {l.split()[0]: l.split() for l in sec2.splitlines() if l.split() and l.split()[0] in ("LARGE", "MID", "SMALL")}
    # LARGE has only 2 rows after the filter -> the stability row is suppressed (needs >= 3)
    assert "LARGE" not in rows
    mid = rows["MID"]
    steady_frac = 2 / 3 * 100                              # M0, M2 steady; M1 erratic
    assert mid[1] == "3" and mid[3] == f"{steady_frac:.0f}%"
    # M2 is steady AND > 15%; M0 (12%) is steady but not high
    assert mid[4] == f"{1 / 3 * 100:.0f}%"
    small = rows["SMALL"]
    assert small[3] == "100%" and small[4] == f"{1 / 3 * 100:.0f}%"      # only S2 (25%) is > 15%


@pandas3_tier_bug
def test_main_piotroski_section_and_correlation(rbl, monkeypatch, tmp_path, capsys):
    strong = dict(Net__Income=[100.0, 60.0, 60.0, 60.0], Operating__Cash__Flow=[150.0, 120.0, 120.0, 120.0],
                  Long__Term__Debt=[50.0, 100.0, 100.0, 100.0], Current__Assets=[600.0, 500.0, 500.0, 500.0],
                  Total__Revenue=[1200.0, 1000.0, 1000.0, 1000.0])
    rocs = default_rocs(L2=flat_ticker(ebit=240, **strong), M2=flat_ticker(ebit=288, **strong),
                        S2=flat_ticker(ebit=200, **strong))
    build_main(rbl, monkeypatch, tmp_path, rocs)
    assert rbl.main() == 0
    out = capsys.readouterr().out
    sec3 = out.split("3. ROCE x PIOTROSKI")[1]
    rows = {l.split()[0]: l.split() for l in sec3.splitlines() if l.split() and l.split()[0] in ("LARGE", "MID", "SMALL")}
    weak_roce = {"LARGE": 15.0, "MID": 18.0, "SMALL": 10.0}
    for t, hi_roce in (("LARGE", 30.0), ("MID", 36.0), ("SMALL", 25.0)):
        r = rows[t]
        assert r[1] == "3" and r[2] == "3.0"           # median F: two weak (3.0) and one strong (7.0)
        assert r[3] == "33%"                           # share with F >= 7
        assert float(r[4].rstrip("%")) == pytest.approx(hi_roce)
        assert float(r[5].rstrip("%")) == pytest.approx(weak_roce[t])     # median ROCE of the two F=3 names
    assert "spearman corr(ROCE, F-score)" in sec3


@pandas3_tier_bug
def test_main_skips_piotroski_section_when_no_f_scores(rbl, monkeypatch, tmp_path, capsys):
    bare = {s: flat_ticker(ebit=100 + 10 * i, years=1) for i, s in enumerate(
        ["L0", "L1", "L2", "M0", "M1", "M2", "S0", "S1", "S2"])}
    build_main(rbl, monkeypatch, tmp_path, bare)
    assert rbl.main() == 0
    assert "3. ROCE x PIOTROSKI" not in capsys.readouterr().out


@pandas3_tier_bug
def test_main_counts_only_symbols_with_statements(rbl, monkeypatch, tmp_path, capsys):
    rocs = default_rocs()
    rocs["S1"] = FakeTicker(boom=True)
    del rocs["M0"]                                      # yfinance constructor raises for it
    build_main(rbl, monkeypatch, tmp_path, rocs)
    assert rbl.main() == 0
    out = capsys.readouterr().out
    assert "statements retrieved for 7/9 (78%)" in out
    csv = pd.read_csv(tmp_path / "reports" / "roace_by_liquidity_india.csv")
    assert set(csv.Symbol) == {"L0", "L1", "L2", "M1", "M2", "S0", "S2"}


@pandas3_tier_bug
def test_main_sampling_is_capped_per_tier(rbl, monkeypatch, tmp_path, capsys, load):
    rbl2 = load("roace_by_liquidity", "2")
    (tmp_path / "reports").mkdir(exist_ok=True)
    build_main(rbl2, monkeypatch, tmp_path, default_rocs())
    assert rbl2.PER_TIER == 2
    assert rbl2.main() == 0
    out = capsys.readouterr().out
    assert "sampling 6 (2/tier)" in out
    csv = pd.read_csv(tmp_path / "reports" / "roace_by_liquidity_india.csv")
    assert csv.tier.value_counts().to_dict() == {"LARGE": 2, "MID": 2, "SMALL": 2}


def test_default_sample_size(load):
    assert load("roace_by_liquidity").PER_TIER == 45


@pytest.mark.xfail(strict=True, raises=KeyError,
                   reason="roace_by_liquidity.py:258 — when no ticker returns statements, "
                          "pd.DataFrame([]).merge(samp, on='Symbol') raises KeyError, so the "
                          "'no data' / return 1 branch at :261 is unreachable")
def test_main_with_no_statements_reports_no_data(rbl, monkeypatch, tmp_path, capsys):
    build_main(rbl, monkeypatch, tmp_path, {s: FakeTicker(boom=True) for s in
                                            ["L0", "L1", "L2", "M0", "M1", "M2", "S0", "S1", "S2"]})
    assert rbl.main() == 1
    assert "no data" in capsys.readouterr().out
