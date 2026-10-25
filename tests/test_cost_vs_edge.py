"""Tests for cost_vs_edge: Corwin-Schultz spread, Amihud ILLIQ, and the net-edge table.

The price panel is a tiny synthetic parquet written to tmp_path; cost_vs_edge.PX is pointed at it."""
import duckdb
import numpy as np
import pandas as pd
import pytest

import cost_vs_edge as C


def _px(tmp_path, rows):
    p = tmp_path / "px.parquet"
    pd.DataFrame(rows).to_parquet(p)
    return str(p)


def test_corwin_schultz_recovers_known_spread(tmp_path, monkeypatch):
    # Constant H=101, L=99 every day: beta = 2 ln(101/99)^2, gamma = ln(101/99)^2,
    # so alpha = ln(101/99) and S = 2(e^a - 1)/(1 + e^a) = 0.02 exactly (the 2% quoted spread).
    d = pd.bdate_range("2019-01-01", periods=260)
    rows = {"Symbol": "A", "Date": d, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e5}
    monkeypatch.setattr(C, "PX", _px(tmp_path, rows))
    cs = C.corwin_schultz(duckdb.connect())
    assert cs.loc[0, "Symbol"] == "A"
    assert cs.loc[0, "spread"] == pytest.approx(0.02, abs=1e-12)


def test_corwin_schultz_requires_200_obs(tmp_path, monkeypatch):
    d = pd.bdate_range("2019-01-01", periods=150)
    rows = {"Symbol": "A", "Date": d, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e5}
    monkeypatch.setattr(C, "PX", _px(tmp_path, rows))
    assert C.corwin_schultz(duckdb.connect()).empty


def test_amihud_hand_computed(tmp_path, monkeypatch):
    # closes alternate 100, 101: |ret| = 1% or 0.990%; dollar volume = close * 1e4
    d = pd.bdate_range("2019-01-01", periods=260)
    close = np.where(np.arange(260) % 2 == 0, 100.0, 101.0)
    rows = {"Symbol": "A", "Date": d, "High": close, "Low": close, "Close": close, "Volume": 1e4}
    monkeypatch.setattr(C, "PX", _px(tmp_path, rows))
    am = C.amihud(duckdb.connect())
    ret = pd.Series(close).pct_change().abs()
    dvol = pd.Series(close * 1e4)
    expect = (ret / dvol).iloc[1:].mean() * 1e6
    assert am.loc[0, "illiq"] == pytest.approx(expect)
    assert am.loc[0, "turnover"] == pytest.approx(np.median(close[1:] * 1e4))   # rows with a return


def _sweep(n_per_tier=20, years=range(2017, 2023), small_edge=False):
    rows = []
    syms = [f"S{i:02d}" for i in range(3 * n_per_tier)]
    for y in years:
        for i, s in enumerate(syms):
            score = float(i % n_per_tier + 1)
            fwd = score / 100 if (small_edge and i < n_per_tier) else 0.10
            rows.append({"Symbol": s, "t": f"{y}-06-01", "fwd_ret": fwd, "canonical": score,
                         "quality": score, "turnaround": score, "safety": score})
    return pd.DataFrame(rows)


def _costs(n):
    syms = [f"S{i:02d}" for i in range(n)]
    cs = pd.DataFrame({"Symbol": syms, "spread": 0.01, "n": 300})
    am = pd.DataFrame({"Symbol": syms, "illiq": 0.0, "turnover": np.arange(1, n + 1) * 1e6})
    return cs, am


def test_gross_edge_is_measured_from_the_sweep_not_hardcoded(tmp_path, monkeypatch, capsys):
    # A sweep where every stock earns the same forward return has ZERO within-tier edge, so the
    # SMALL best gross edge reported must be +0.0% -- whatever an earlier run measured.
    p = tmp_path / "sweep.csv"
    _sweep().to_csv(p, index=False)
    cs, am = _costs(60)
    monkeypatch.setattr(C, "SWEEP", str(p))
    monkeypatch.setattr(C, "corwin_schultz", lambda con: cs)
    monkeypatch.setattr(C, "amihud", lambda con: am)
    assert C.main() == 0
    out = capsys.readouterr().out
    assert "best gross +0.0%" in out, out[-600:]
    assert "best gross +13.3%" not in out


def test_within_tier_edge_hand_computed():
    # SMALL tier: fwd_ret = score/100 for scores 1..20 -> top-10 median 15.5%, tier median 10.5%
    d = _sweep(small_edge=True)
    tiers = pd.DataFrame({"Symbol": [f"S{i:02d}" for i in range(60)],
                          "tier": ["SMALL"] * 20 + ["MID"] * 20 + ["LARGE"] * 20})
    g = C.within_tier_gross(d, tiers, ["canonical", "safety"], top_n=10)
    assert g["SMALL"]["canonical"] == pytest.approx(5.0)
    assert g["MID"]["safety"] == pytest.approx(0.0)


def test_within_tier_edge_needs_enough_years():
    d = _sweep(years=range(2020, 2023))           # 3 rebalances < 4 required by the sweep
    tiers = pd.DataFrame({"Symbol": [f"S{i:02d}" for i in range(60)], "tier": ["SMALL"] * 60})
    g = C.within_tier_gross(d, tiers, ["canonical"], top_n=10)
    assert np.isnan(g["SMALL"]["canonical"])
