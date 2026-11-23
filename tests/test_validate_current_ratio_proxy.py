import datetime as dt

import numpy as np
import pandas as pd
import pytest
from helpers import FakeTicker, install_fake_yfinance, stmt, write_parquet

YEARS6 = [dt.date(2020 + i, 3, 31) for i in range(6)]


@pytest.fixture
def vcr(load, tmp_path, monkeypatch):
    mod = load("validate_current_ratio_proxy", "60")
    (tmp_path / "reports").mkdir()
    monkeypatch.chdir(tmp_path)
    return mod


def timestamp_keys(vcr, monkeypatch):
    """main() looks years up with the Timestamps duckdb's .df() returns, while truth() keys its
    dict with datetime.date, so unpatched every lookup misses (see the xfail test at the end).
    The logic tests below put truth() on Timestamp keys so the rest of main() can be exercised."""
    real = vcr.truth

    def wrapped(sym):
        r = real(sym)
        if r:
            r = {"Symbol": r["Symbol"], "cr": {pd.Timestamp(k): v for k, v in r["cr"].items()}}
        return r
    monkeypatch.setattr(vcr, "truth", wrapped)


def levels(dirs, start=1.0):
    out = [start]
    for d in dirs:
        out.append(out[-1] * (1.1 if d > 0 else 1 / 1.1))
    return out


def world(vcr, tmp_path, monkeypatch, spec, extra_rows=None, patch=True):
    """spec: {ticker: (proxy_dirs, true_dirs)} with len(dirs)+1 fiscal years each."""
    rows, tickers = [], {}
    for tk, (pdirs, tdirs) in spec.items():
        years = YEARS6[:len(pdirs) + 1]
        pl, tl = levels(pdirs), levels(tdirs)
        for y, p in zip(years, pl):
            rows.append({"ticker": tk, "fy_end": pd.Timestamp(y), "receivables": 60 * p,
                         "inventory": 30 * p, "cash": 10 * p, "other_liab": 100.0})
        idx = [pd.Timestamp(y) for y in reversed(years)]
        ca = [100 * v for v in reversed(tl)]
        tickers[f"{tk}.NS"] = FakeTicker(balance_sheet=stmt(
            {"Current Assets": ca, "Current Liabilities": [100.0] * len(ca)}, idx))
    rows += extra_rows or []
    vcr.FUND = write_parquet(pd.DataFrame(rows), tmp_path / "f.parquet")
    install_fake_yfinance(monkeypatch, tickers)
    if patch:
        timestamp_keys(vcr, monkeypatch)


UP, DN = 1, -1


def agree(n_agree, n_total, tk_prefix="T"):
    """Spec with n_total company-years of which n_agree have matching direction (5 pairs/ticker)."""
    spec, k, left = {}, 0, n_total
    while left > 0:
        m = min(5, left)
        pd_, td_ = [], []
        for _ in range(m):
            ok = n_agree > 0
            n_agree -= 1 if ok else 0
            pd_.append(UP)
            td_.append(UP if ok else DN)
        spec[f"{tk_prefix}{k}"] = (pd_, td_)
        left -= m
        k += 1
    return spec


@pytest.mark.parametrize("n_agree,verdict", [
    (16, "SOUND -> use across the 10y series"),               # 80.0% exactly
    (15, "MARGINAL -> usable only if flagged approximate"),   # 75%
    (13, "MARGINAL -> usable only if flagged approximate"),   # 65.0% exactly
    (12, "DROP test 6"),                                      # 60%
])
def test_verdict_thresholds(vcr, tmp_path, monkeypatch, capsys, n_agree, verdict):
    world(vcr, tmp_path, monkeypatch, agree(n_agree, 20))
    assert vcr.main() == 0
    out = capsys.readouterr().out
    pct = n_agree / 20 * 100
    assert f"proxy delta sign matches truth: {pct:.1f}%" in out
    assert f"VERDICT: {pct:.1f}% -> {verdict}" in out
    assert "n = 20 company-years across 4 tickers" in out


def test_levels_and_majority_baseline(vcr, tmp_path, monkeypatch, capsys):
    spec = {"A": ([UP] * 4, [UP, UP, UP, DN]), "B": ([UP] * 2, [UP, DN])}
    world(vcr, tmp_path, monkeypatch, spec)
    assert vcr.main() == 0
    out = capsys.readouterr().out
    # true rising: 3 of 4 + 1 of 2 = 4 of 6 -> majority baseline 66.7%
    assert "always-guess-the-majority baseline: 66.7%" in out
    assert "n = 6 company-years across 2 tickers" in out
    assert "yfinance returned true current ratios for 2/2" in out
    d = pd.read_csv(tmp_path / "reports" / "current_ratio_proxy_validation.csv")
    assert len(d) == 6 and list(d.columns) == ["Symbol", "fy", "true_cr", "proxy_cr", "true_rising", "proxy_rising"]
    a = d[d.Symbol == "A"].sort_values("fy")
    assert a.proxy_cr.tolist() == pytest.approx(levels([UP] * 4)[1:])   # proxy level = (rec+inv+cash)/other_liab
    assert a.true_cr.tolist() == pytest.approx(levels([UP, UP, UP, DN])[1:])
    assert "median proxy current ratio" in out and "median true current ratio" in out


def test_baseline_is_the_larger_side(vcr, tmp_path, monkeypatch, capsys):
    world(vcr, tmp_path, monkeypatch, {"A": ([UP] * 4, [DN] * 4)})
    vcr.main()
    out = capsys.readouterr().out
    assert "always-guess-the-majority baseline: 100.0%" in out


def test_only_top_n_tickers_by_history_length_are_tested(vcr, tmp_path, monkeypatch, capsys):
    vcr.N = 2
    spec = {"LONG": ([UP] * 5, [UP] * 5), "MID": ([UP] * 4, [UP] * 4), "SHORT": ([UP] * 2, [UP] * 2)}
    world(vcr, tmp_path, monkeypatch, spec)
    assert vcr.main() == 0
    out = capsys.readouterr().out
    assert "testing 2 tickers against yfinance" in out
    d = pd.read_csv(tmp_path / "reports" / "current_ratio_proxy_validation.csv")
    assert set(d.Symbol) == {"LONG", "MID"}


def test_default_n_is_sixty(load):
    assert load("validate_current_ratio_proxy").N == 60
    assert load("validate_current_ratio_proxy", "7").N == 7


def test_rows_with_bad_other_liab_or_null_receivables_are_ignored(vcr, tmp_path, monkeypatch, capsys):
    bad = [{"ticker": "A", "fy_end": pd.Timestamp(dt.date(2026, 3, 31)), "receivables": 60.0,
            "inventory": 30.0, "cash": 10.0, "other_liab": 0.0},
           {"ticker": "A", "fy_end": pd.Timestamp(dt.date(2027, 3, 31)), "receivables": None,
            "inventory": 30.0, "cash": 10.0, "other_liab": 100.0}]
    world(vcr, tmp_path, monkeypatch, {"A": ([UP] * 2, [UP] * 2)}, extra_rows=bad)
    vcr.main()
    d = pd.read_csv(tmp_path / "reports" / "current_ratio_proxy_validation.csv")
    assert len(d) == 2


def test_missing_inventory_or_cash_counts_as_zero(vcr, tmp_path, monkeypatch, capsys):
    world(vcr, tmp_path, monkeypatch, {"A": ([UP] * 2, [UP] * 2)})
    import duckdb
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{vcr.FUND}'").df()
    df["inventory"] = None
    df["cash"] = None
    vcr.FUND = write_parquet(df, tmp_path / "f2.parquet")
    assert vcr.main() == 0
    d = pd.read_csv(tmp_path / "reports" / "current_ratio_proxy_validation.csv")
    assert d.proxy_cr.tolist() == pytest.approx([(60 * v) / 100 for v in levels([UP] * 2)[1:]])


def test_no_overlapping_years_returns_one(vcr, tmp_path, monkeypatch, capsys):
    world(vcr, tmp_path, monkeypatch, {"A": ([UP] * 2, [UP] * 2)})
    install_fake_yfinance(monkeypatch, {"A.NS": FakeTicker(balance_sheet=stmt(
        {"Current Assets": [1.0], "Current Liabilities": [1.0]}, ["1999-03-31"]))})
    assert vcr.main() == 1
    assert "no overlapping years — cannot validate" in capsys.readouterr().out


# ── truth() ──────────────────────────────────────────────────────────────────
def truth(vcr, monkeypatch, ticker):
    install_fake_yfinance(monkeypatch, {"X.NS": ticker})
    return vcr.truth("X")


def test_truth_computes_current_ratio_per_year(vcr, monkeypatch):
    t = FakeTicker(balance_sheet=stmt({"Total Current Assets": [300.0, 200.0], "Total Current Liabilities": [100.0, 100.0]},
                                      ["2025-03-31", "2024-03-31"]))
    r = truth(vcr, monkeypatch, t)
    assert r == {"Symbol": "X", "cr": {dt.date(2025, 3, 31): 3.0, dt.date(2024, 3, 31): 2.0}}


def test_truth_skips_zero_liability_years_and_unmatched_dates(vcr, monkeypatch):
    t = FakeTicker(balance_sheet=stmt({"Current Assets": [300.0, 200.0, 50.0], "Current Liabilities": [0.0, 100.0, None]},
                                      ["2025-03-31", "2024-03-31", "2023-03-31"]))
    assert truth(vcr, monkeypatch, t)["cr"] == {dt.date(2024, 3, 31): 2.0}
    only_assets = FakeTicker(balance_sheet=stmt({"Current Assets": [300.0]}, ["2025-03-31"]))
    assert truth(vcr, monkeypatch, only_assets) is None
    all_zero = FakeTicker(balance_sheet=stmt({"Current Assets": [300.0], "Current Liabilities": [0.0]}, ["2025-03-31"]))
    assert truth(vcr, monkeypatch, all_zero) is None


@pytest.mark.parametrize("bs", [None, pd.DataFrame()])
def test_truth_empty_balance_sheet(vcr, monkeypatch, bs):
    assert truth(vcr, monkeypatch, FakeTicker(balance_sheet=bs)) is None


def test_truth_fetch_errors(vcr, monkeypatch):
    assert truth(vcr, monkeypatch, FakeTicker(boom=True)) is None
    install_fake_yfinance(monkeypatch, {})
    assert vcr.truth("NOPE") is None


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="validate_current_ratio_proxy.py:89/119 — truth() keys years with datetime.date "
                          "but main() looks them up with the pandas Timestamps duckdb's .df() returns, so "
                          "t['cr'].get(...) always misses and main() always ends 'no overlapping years' (rc 1)")
def test_unpatched_main_matches_years_between_proxy_and_truth(vcr, tmp_path, monkeypatch, capsys):
    world(vcr, tmp_path, monkeypatch, agree(16, 20), patch=False)
    assert vcr.main() == 0
    assert "n = 20 company-years across 4 tickers" in capsys.readouterr().out
