"""Tests for adaptive_liquidity floors, gate, tiers and capacity."""
import pandas as pd
import pytest

import adaptive_liquidity as A


def test_floors():
    assert A.market_floor("IN") == 120_000
    assert A.market_floor("india") == 120_000      # first two letters
    assert A.market_floor("US") == 0.0
    assert A.market_floor(None) == 0.0
    assert A.scan_floor("US") == A.STRUCTURAL_FLOOR_USD
    assert A.scan_floor("IN") == 120_000


def test_gate_min_adv():
    g = A.Gate("US", capital_usd=1_000_000, positions=10)
    assert g.position_usd == 100_000
    assert g.min_adv_usd == pytest.approx(100_000 / 0.15)
    # India: tiny capital -> the Rs 1cr policy floor binds
    assert A.Gate("IN", 10_000, 10).min_adv_usd == 120_000
    assert A.Gate("US", 1_000, 0).position_usd == 1_000      # positions guarded


def test_classify_tradeable_and_tiers():
    df = pd.DataFrame({"adv_usd": [1e3 * (i + 1) for i in range(20)] + [0.0]})
    g = A.Gate("US", capital_usd=10_000, positions=10)     # position 1,000 -> min ADV 6,667
    out = A.classify(df, g)
    assert (out.loc[out.adv_usd >= 7_000, "tradeable"]).all()
    assert not out.loc[out.adv_usd <= 6_000, "tradeable"].any()
    assert out.loc[20, "adv_pct"] == float("inf")
    t = out[out.tradeable]
    assert len(t) == 14 and t["tier"].notna().all()
    assert t.loc[t.adv_usd.idxmax(), "tier"] == "T1_MOST_LIQUID"
    assert t.loc[t.adv_usd.idxmin(), "tier"] == "T5_MOST_ILLIQUID"


def test_capacity_hand_computed():
    df = pd.DataFrame({"adv_usd": [100.0, 200.0, 300.0, 400.0, 500.0]})
    c = A.capacity(df, positions=10).set_index("universe_point")
    assert c.loc["median name", "max_position_usd"] == pytest.approx(300 * 0.15)
    assert c.loc["median name", "max_portfolio_usd"] == pytest.approx(300 * 0.15 * 10)


def test_retier_keeps_unknown_and_small_frames():
    df = pd.DataFrame({"Median_Turnover": list(range(1, 26)) + [None],
                       "Liquidity_Tier": ["X"] * 25 + ["UNKNOWN"]})
    out = A.retier(df)
    assert out.loc[25, "Liquidity_Tier"] == "UNKNOWN"
    assert out.loc[24, "Liquidity_Tier"] == "T1_MOST_LIQUID"
    assert out.loc[0, "Liquidity_Tier"] == "T5_MOST_ILLIQUID"
    small = df.head(5)
    assert A.retier(small) is small
