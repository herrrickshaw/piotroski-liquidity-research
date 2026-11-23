import pandas as pd
import pytest

import adaptive_liquidity as AL


def test_market_floor_normalises_the_market_code():
    inr = AL.MARKET_FLOOR_USD["IN"]
    for m in ("IN", "in", " in ", "India", "IND"):
        assert AL.market_floor(m) == inr
    for m in ("US", "", None, "X"):
        assert AL.market_floor(m) == AL.DEFAULT_FLOOR_USD == 0.0
    assert AL.ABSOLUTE_FLOOR_USD == inr


def test_scan_floor_is_structural_unless_policy_is_higher():
    assert AL.scan_floor("US") == AL.STRUCTURAL_FLOOR_USD
    assert AL.scan_floor("IN") == max(AL.STRUCTURAL_FLOOR_USD, AL.MARKET_FLOOR_USD["IN"])
    assert AL.scan_floor("IN") > AL.STRUCTURAL_FLOOR_USD


def test_gate_position_size_and_min_adv():
    g = AL.Gate("US", capital_usd=1_500_000, positions=10)
    assert g.position_usd == 150_000
    assert g.min_adv_usd == pytest.approx(150_000 / (AL.MAX_ADV_PCT / 100))
    # positions=0 must not divide by zero: treated as 1
    assert AL.Gate("US", 1000, 0).position_usd == 1000
    # market policy floor wins when capital is small, capital floor wins when large
    assert AL.Gate("IN", 1_000, 10).min_adv_usd == AL.MARKET_FLOOR_USD["IN"]
    big = AL.Gate("IN", 1_000_000_000, 10)
    assert big.min_adv_usd == pytest.approx(1e8 / 0.15) and big.min_adv_usd > AL.MARKET_FLOOR_USD["IN"]


def universe(n=40, step=1_000_000):
    return pd.DataFrame({"sym": [f"S{i}" for i in range(n)],
                         "adv_usd": [step * (i + 1) for i in range(n)]})


def test_classify_adds_tradeable_adv_pct_and_percentile_tiers():
    df = universe()
    g = AL.Gate("US", capital_usd=1_500_000, positions=10)   # position 150k, floor 1.0M ADV
    out = AL.classify(df, g)
    assert list(df.columns) == ["sym", "adv_usd"]              # input untouched
    assert out.loc[0, "adv_pct"] == pytest.approx(15.0)       # 150k / 1M
    assert out.loc[0, "tradeable"]                             # exactly at both limits -> tradeable
    assert out.tradeable.all()
    assert out.loc[39, "adv_pct"] == pytest.approx(150_000 / 40_000_000 * 100)
    assert out.loc[39, "tier"] == "T1_MOST_LIQUID"
    assert out.loc[0, "tier"] == "T5_MOST_ILLIQUID"
    counts = out.tier.value_counts()
    assert counts.to_dict() == {"T1_MOST_LIQUID": 9, "T2_LIQUID": 8, "T3_MID": 8,
                                "T4_ILLIQUID": 8, "T5_MOST_ILLIQUID": 7}


def test_classify_untradeable_names_get_no_tier_and_do_not_define_percentiles():
    df = pd.DataFrame({"adv_usd": [0, -5, 10, 500_000] + [2_000_000 * (i + 1) for i in range(12)]})
    g = AL.Gate("US", capital_usd=1_500_000, positions=10)
    out = AL.classify(df, g)
    assert out.loc[0, "adv_pct"] == float("inf") and out.loc[1, "adv_pct"] == float("inf")
    assert not out.loc[:3, "tradeable"].any()
    assert out.loc[:3, "tier"].isna().all()
    assert out.loc[4:, "tradeable"].all()
    # ranks are over the 12 tradeable names only -> the smallest of them is the bottom tier
    assert out.loc[4, "tier"] == "T5_MOST_ILLIQUID"
    assert out.loc[15, "tier"] == "T1_MOST_LIQUID"


def test_classify_needs_ten_tradeable_names_for_tiers():
    g = AL.Gate("US", 1_500_000, 10)
    nine = pd.DataFrame({"adv_usd": [2_000_000.0] * 9})
    assert AL.classify(nine, g).tier.isna().all()
    ten = pd.DataFrame({"adv_usd": [2_000_000.0 * (i + 1) for i in range(10)]})
    assert AL.classify(ten, g).tier.notna().all()


def test_classify_market_floor_gates_even_when_capital_is_tiny():
    floor = AL.MARKET_FLOOR_USD["IN"]
    df = pd.DataFrame({"adv_usd": [floor - 1, floor, floor + 1]})
    out = AL.classify(df, AL.Gate("IN", 100, 10))
    assert list(out.tradeable) == [False, True, True]


def test_classify_custom_adv_column():
    df = pd.DataFrame({"turn": [5_000_000.0] * 12})
    out = AL.classify(df, AL.Gate("US", 100_000, 10), adv_col="turn")
    assert out.tradeable.all() and out.adv_pct.iloc[0] == pytest.approx(10_000 / 5_000_000 * 100)


def test_capacity_scales_with_adv_ceiling_and_positions():
    df = pd.DataFrame({"adv_usd": list(range(1, 101))})
    c = AL.capacity(df, positions=4)
    assert list(c.universe_point) == ["median name", "bottom quartile", "bottom decile"]
    for _, r in c.iterrows():
        assert r.max_position_usd == pytest.approx(r.adv_usd * AL.MAX_ADV_PCT / 100)
        assert r.max_portfolio_usd == pytest.approx(r.max_position_usd * 4)
    assert c.adv_usd[0] == df.adv_usd.quantile(0.5)
    assert c.adv_usd[1] == df.adv_usd.quantile(0.25)
    assert c.adv_usd[2] == df.adv_usd.quantile(0.10)
    assert AL.capacity(df).max_portfolio_usd[0] == pytest.approx(c.max_portfolio_usd[0] / 4 * 10)


def scan_frame(n=30):
    return pd.DataFrame({"Median_Turnover": [float(i + 1) for i in range(n)],
                         "Liquidity_Tier": ["ABS"] * n})


def test_retier_replaces_absolute_tiers_with_percentile_tiers():
    df = scan_frame(30)
    out = AL.retier(df)
    assert (df.Liquidity_Tier == "ABS").all()                  # input not mutated
    assert out.Liquidity_Tier.iloc[-1] == "T1_MOST_LIQUID"
    assert out.Liquidity_Tier.iloc[0] == "T5_MOST_ILLIQUID"
    assert set(out.Liquidity_Tier) == {n for _, n in AL.TIER_CUTS}


def test_retier_leaves_unknown_rows_and_unrankable_rows_alone():
    df = scan_frame(30)
    df.loc[0, "Liquidity_Tier"] = "UNKNOWN"
    df.loc[1, "Median_Turnover"] = float("nan")
    df.loc[2, "Median_Turnover"] = 0.0
    out = AL.retier(df)
    assert out.loc[0, "Liquidity_Tier"] == "UNKNOWN"
    assert out.loc[1, "Liquidity_Tier"] == "ABS" and out.loc[2, "Liquidity_Tier"] == "ABS"
    assert out.loc[3:, "Liquidity_Tier"].isin([n for _, n in AL.TIER_CUTS]).all()


def test_retier_needs_twenty_rankable_rows_and_the_columns():
    small = scan_frame(19)
    assert AL.retier(small) is small
    assert (AL.retier(scan_frame(20)).Liquidity_Tier != "ABS").all()
    no_col = pd.DataFrame({"x": [1]})
    assert AL.retier(no_col) is no_col
    only_one = pd.DataFrame({"Median_Turnover": [1.0] * 25})
    assert AL.retier(only_one) is only_one


def test_retier_custom_columns():
    df = pd.DataFrame({"t": [float(i + 1) for i in range(25)], "tier": ["A"] * 25})
    out = AL.retier(df, turnover_col="t", tier_col="tier")
    assert out.tier.iloc[-1] == "T1_MOST_LIQUID"


def test_describe_formats_one_line_per_market():
    df = pd.DataFrame({"adv_usd": [0, 1_000_000.0, 3_000_000.0, 5_000_000.0]})
    s = AL.describe(df, "IN")
    assert s.startswith("  IN ")
    assert "n=     3" in s
    assert "ADV median $    3.00M" in s
    assert "above Rs1cr floor:  100%" in s
    assert AL.describe(pd.DataFrame({"adv_usd": [0, 0.0]}), "US") == "  US: no data"
    low = pd.DataFrame({"adv_usd": [AL.ABSOLUTE_FLOOR_USD - 1, AL.ABSOLUTE_FLOOR_USD]})
    assert "above Rs1cr floor:   50%" in AL.describe(low, "IN")
