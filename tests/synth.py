"""Deterministic synthetic panels + fake yfinance Ticker for offline tests.

No network, no tracked-data dependency: everything is generated from a fixed seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_TICKERS = 140


def _prices(symbols, start, end, seed, volume=True):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    frames = []
    for i, s in enumerate(symbols):
        r = rng.normal(0.0004, 0.02, len(dates))
        close = 20 * np.exp(np.cumsum(r)) * (1 + i / 50)
        if i == 3:                       # a unit split: filtered as a corporate action
            close[len(close) // 2:] = close[len(close) // 2:] / 100
        df = pd.DataFrame({"Date": dates, "Symbol": s, "Close": close})
        if volume:
            df["Volume"] = rng.integers(1_000, 500_000, len(dates)) * (1 + (i % 7))
            df["High"] = close * 1.01
            df["Low"] = close * 0.99
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def us_panels(tmp_dir):
    """Write synthetic US fundamentals + prices parquet; return (fund_path, px_path)."""
    rng = np.random.default_rng(11)
    rows = []
    syms = [f"T{i:03d}" for i in range(N_TICKERS)]
    for i, s in enumerate(syms):
        base = rng.uniform(1e8, 5e9)
        for y in range(2013, 2026):
            ta = base * (1 + 0.05 * (y - 2013)) * rng.uniform(0.9, 1.1)
            rev = ta * rng.uniform(0.4, 1.4)
            rows.append({
                "ticker": s, "fy_end": f"{y}-12-31",
                "filed": (pd.Timestamp(f"{y}-12-31") + pd.Timedelta(days=int(rng.integers(30, 120)))).strftime("%Y-%m-%d"),
                "ebit": ta * rng.uniform(-0.02, 0.25),
                "net_income": ta * rng.uniform(-0.05, 0.12),
                "cfo": ta * rng.uniform(-0.02, 0.15),
                "total_assets": ta,
                "current_assets": ta * rng.uniform(0.2, 0.5),
                "current_liabilities": ta * rng.uniform(0.1, 0.3),
                "long_term_debt": np.nan if i % 3 == 0 else ta * rng.uniform(0.05, 0.4),
                "shares": 1e7 * rng.uniform(0.97, 1.05),
                "revenue": rev,
                "gross_profit": np.nan if i % 4 == 0 else rev * rng.uniform(0.2, 0.6),
                "equity": ta * 0.4,
            })
    fund = pd.DataFrame(rows)
    fp, pp = f"{tmp_dir}/US_fund.parquet", f"{tmp_dir}/US_px.parquet"
    fund.to_parquet(fp)
    _prices(syms, "2016-01-01", "2026-07-10", 5).to_parquet(pp)
    return fp, pp


def in_panels(tmp_dir):
    """Write synthetic India fundamentals + prices parquet; return (fund_path, px_path)."""
    rng = np.random.default_rng(13)
    rows = []
    syms = [f"IN{i:03d}" for i in range(N_TICKERS)]
    for i, s in enumerate(syms):
        base = rng.uniform(1e3, 5e4)
        for y in range(2012, 2026):
            ta = base * (1 + 0.06 * (y - 2012)) * rng.uniform(0.9, 1.1)
            rev = ta * rng.uniform(0.4, 1.4)
            ce = ta * rng.uniform(0.4, 0.8)
            ebit = ta * rng.uniform(-0.02, 0.25)
            mfr = i % 2 == 0
            rows.append({
                "ticker": s, "fy_end": f"{y}-03-31",
                "net_income": ta * rng.uniform(-0.05, 0.12), "revenue": rev,
                "cfo": np.nan if i % 11 == 0 else ta * rng.uniform(-0.02, 0.15),
                "shares": 1e7 * rng.uniform(0.97, 1.05),
                "equity_capital": ta * 0.1, "reserves": ta * 0.3,
                "borrowings": ta * rng.uniform(0.05, 0.4), "other_liab": ta * 0.2,
                "receivables": ta * 0.1, "inventory": ta * 0.1, "cash": ta * 0.05,
                "ebit": ebit, "capital_employed": ce, "roce": ebit / ce,
                "raw_material_cost": rev * rng.uniform(0.35, 0.6) if mfr else rev * 0.05,
                "change_in_inventory": rng.uniform(-100, 100),
                "power_and_fuel": rev * 0.03, "other_mfr_exp": rev * 0.05,
            })
    fund = pd.DataFrame(rows)
    fp, pp = f"{tmp_dir}/IN_fund.parquet", f"{tmp_dir}/IN_px.parquet"
    fund.to_parquet(fp)
    _prices(syms, "2017-01-01", "2026-07-10", 6, volume=False).to_parquet(pp)
    return fp, pp


class FakeTicker:
    """Minimal stand-in for yfinance.Ticker exposing the three statements."""

    def __init__(self, inc, bal, cfs):
        self.income_stmt, self.balance_sheet, self.cashflow = inc, bal, cfs


def fake_ticker(seed=0, drop=()):
    rng = np.random.default_rng(seed)
    yrs = pd.to_datetime(["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31", "2021-03-31"])

    def frame(d):
        return pd.DataFrame({k: v for k, v in d.items() if k not in drop}, index=yrs).T

    ta = np.array([1000., 950, 900, 860, 800]) * rng.uniform(0.9, 1.1)
    inc = frame({
        "EBIT": ta * rng.uniform(0.05, 0.25, 5), "Net Income": ta * rng.uniform(0.0, 0.12, 5),
        "Total Revenue": ta * rng.uniform(0.6, 1.3, 5),
        "Gross Profit": ta * rng.uniform(0.2, 0.5, 5)})
    bal = frame({
        "Total Assets": ta, "Current Liabilities": ta * rng.uniform(0.1, 0.3, 5),
        "Current Assets": ta * rng.uniform(0.3, 0.5, 5),
        "Long Term Debt": ta * rng.uniform(0.05, 0.3, 5),
        "Ordinary Shares Number": 1e6 * rng.uniform(0.98, 1.02, 5),
        "Cash And Cash Equivalents": ta * 0.05})
    cfs = frame({"Operating Cash Flow": ta * rng.uniform(0.0, 0.15, 5)})
    return FakeTicker(inc, bal, cfs)
