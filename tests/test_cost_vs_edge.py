import math
import re

import duckdb
import numpy as np
import pandas as pd
import pytest
from helpers import bdays, walk, write_parquet

TIERS = {"S": dict(a=0.01, vol=2_000), "M": dict(a=0.005, vol=200_000), "L": dict(a=0.001, vol=2_000_000)}


@pytest.fixture
def cve(load):
    return load("cost_vs_edge")


def symbol_frame(name, a, vol, n=260, seed=1, start="2019-01-02", close0=10.0):
    close = walk(n, seed, close0, 0.015)
    return pd.DataFrame({"Date": bdays(start, n), "Symbol": name, "Close": close,
                         "High": 100 * (1 + a), "Low": 100 * (1 - a),
                         "Volume": float(vol)})


def universe(n=260):
    frames = []
    for ti, (t, s) in enumerate(TIERS.items()):
        for k in range(3):
            frames.append(symbol_frame(f"{t}{k}", s["a"], s["vol"] * (1 + 0.1 * k), n,
                                       seed=10 * ti + k + 1))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def world(cve, tmp_path, monkeypatch):
    px = universe()
    cve.PX = write_parquet(px, tmp_path / "px.parquet")
    names = sorted(px.Symbol.unique())
    sweep = pd.DataFrame({"Symbol": names + names[:2] + ["NOPX"], "canonical": 1.0})
    (tmp_path / "reports").mkdir()
    sweep.to_csv(tmp_path / "reports" / "sweep_piotroski_plus_us.csv", index=False)
    monkeypatch.chdir(tmp_path)
    return px


def test_corwin_schultz_constant_range_equals_twice_the_half_range(cve, world):
    cs = cve.corwin_schultz(duckdb.connect()).set_index("Symbol")
    assert set(cs.index) == set(world.Symbol)
    for sym, row in cs.iterrows():
        a = TIERS[sym[0]]["a"]
        # H = P(1+a), L = P(1-a) every day -> S = 2a exactly
        assert row.spread == pytest.approx(2 * a, rel=1e-9)
        assert row.n == 259                         # one pair per bar except the last


def test_corwin_schultz_matches_the_published_formula_on_a_moving_range(cve, tmp_path):
    rng = np.random.RandomState(5)
    n = 260
    base = 100 + np.cumsum(rng.normal(0, 0.5, n))
    lo = base - rng.uniform(0.5, 1.5, n)
    hi = base + rng.uniform(0.5, 1.5, n)
    df = pd.DataFrame({"Date": bdays("2019-01-02", n), "Symbol": "Z", "Close": base,
                       "High": hi, "Low": lo, "Volume": 1e5})
    cve.PX = write_parquet(df, tmp_path / "p.parquet")
    got = cve.corwin_schultz(duckdb.connect()).iloc[0]
    k = 3 - 2 * math.sqrt(2)
    spreads = []
    for i in range(n - 1):
        beta = math.log(hi[i] / lo[i]) ** 2 + math.log(hi[i + 1] / lo[i + 1]) ** 2
        gamma = math.log(max(hi[i], hi[i + 1]) / min(lo[i], lo[i + 1])) ** 2
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
        spreads.append(max(2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha)), 0))
    assert got.spread == pytest.approx(float(np.median(spreads)))
    assert got.n == n - 1
    assert 0 < got.spread < 1


def test_corwin_schultz_filters(cve, tmp_path):
    ok = symbol_frame("OK200", 0.01, 1e4, n=201)       # 200 pairs -> exactly the minimum
    few = symbol_frame("FEW199", 0.01, 1e4, n=200)     # 199 pairs
    old = symbol_frame("OLD", 0.01, 1e4, n=260, start="2016-01-04")
    inv = symbol_frame("INV", 0.01, 1e4, n=260)
    inv["High"], inv["Low"] = inv["Low"], inv["High"]  # High < Low is rejected
    zero = symbol_frame("ZERO", 0.01, 1e4, n=260)
    zero["Low"] = 0.0
    cve.PX = write_parquet(pd.concat([ok, few, old, inv, zero]), tmp_path / "p.parquet")
    cs = cve.corwin_schultz(duckdb.connect()).set_index("Symbol")
    assert list(cs.index) == ["OK200"] and cs.n["OK200"] == 200


def test_corwin_schultz_floors_negative_estimates_at_zero(cve, tmp_path):
    # a day whose range is tiny next to the neighbours' combined range gives alpha < 0
    n = 260
    hi = np.full(n, 100.5); lo = np.full(n, 99.5)
    hi[1::2] = 110.0; lo[1::2] = 90.0          # alternating wide / narrow days -> noisy estimates
    df = pd.DataFrame({"Date": bdays("2019-01-02", n), "Symbol": "N", "Close": 100.0,
                       "High": hi, "Low": lo, "Volume": 1e5})
    cve.PX = write_parquet(df, tmp_path / "p.parquet")
    assert cve.corwin_schultz(duckdb.connect()).spread.iloc[0] >= 0.0


def test_amihud_is_mean_abs_return_per_dollar_traded(cve, world):
    am = cve.amihud(duckdb.connect()).set_index("Symbol")
    for sym in ("S0", "M2", "L1"):
        d = world[world.Symbol == sym].sort_values("Date")
        ret = d.Close.pct_change().iloc[1:]
        dvol = (d.Close * d.Volume).iloc[1:]
        assert am.illiq[sym] == pytest.approx((ret.abs() / dvol).mean() * 1e6)
        assert am.turnover[sym] == pytest.approx(dvol.median())
    assert am.illiq["S0"] > am.illiq["M0"] > am.illiq["L0"]    # thinner -> more impact
    assert am.turnover["S0"] < am.turnover["M0"] < am.turnover["L0"]


def test_amihud_filters(cve, tmp_path):
    ok = symbol_frame("OK200", 0.01, 1e4, n=201)
    few = symbol_frame("FEW199", 0.01, 1e4, n=200)
    nov = symbol_frame("NOVOL", 0.01, 0.0, n=260)
    old = symbol_frame("OLD", 0.01, 1e4, n=260, start="2016-01-04")
    cve.PX = write_parquet(pd.concat([ok, few, nov, old]), tmp_path / "p.parquet")
    am = cve.amihud(duckdb.connect())
    assert list(am.Symbol) == ["OK200"]


def run(cve, capsys):
    rc = cve.main()
    return rc, capsys.readouterr().out


def test_main_coverage_line_counts_tested_symbols_with_estimates(cve, world, capsys):
    rc, out = run(cve, capsys)
    assert rc == 0
    # 9 symbols have both estimates; the sweep also lists NOPX (no prices) -> 9 of 10 tested
    assert "cost estimates for 9 of 10 tested symbols" in out
    assert "NOT investment advice" in out


def test_main_cost_inputs_by_tier(cve, world, capsys):
    _, out = run(cve, capsys)
    block = out.split("COST INPUTS BY TIER")[1].split("NET EDGE")[0]
    rows = {l.split()[0]: l.split() for l in block.splitlines() if l.split() and l.split()[0] in ("SMALL", "MID", "LARGE")}
    assert [int(rows[t][1]) for t in ("SMALL", "MID", "LARGE")] == [3, 3, 3]
    assert rows["SMALL"][3] == "2.00%" and rows["MID"][3] == "1.00%" and rows["LARGE"][3] == "0.20%"


def test_main_net_edge_table_arithmetic(cve, world, capsys):
    _, out = run(cve, capsys)
    am = cve.amihud(duckdb.connect()).set_index("Symbol")
    small = am.loc[["S0", "S1", "S2"]]
    illiq, turn = small.illiq.median(), small.turnover.median()
    sec = out.split("--- $10,000 portfolio")[1].split("--- $50,000")[0]
    pos = 10_000 / cve.PORTFOLIO_N
    cost = (0.02 + 2 * illiq * pos / 1e6) * 100
    adv = pos / turn * 100
    row = [l for l in sec.splitlines() if l.strip().startswith("SMALL")][0].split()
    assert row[1] == f"{cost:.2f}%" and row[2] == f"{adv:.1f}%"
    gross = {"canonical": 5.7, "quality": 6.5, "turnaround": 13.3, "safety": 7.1}
    assert row[3:7] == [f"{g - cost:+.1f}%" for g in gross.values()]
    assert "(gross +5.7) (gross +6.5) (gross +13.3) (gross +7.1)" in sec


def test_main_verdict_classifies_every_capital(cve, world, capsys):
    _, out = run(cve, capsys)
    am = cve.amihud(duckdb.connect()).set_index("Symbol")
    small = am.loc[["S0", "S1", "S2"]]
    illiq, turn = small.illiq.median(), small.turnover.median()
    verdict = out.split("=== VERDICT ===")[1]
    lines = [l for l in verdict.splitlines() if l.strip().startswith("$")]
    assert len(lines) == len(cve.CAPITALS)
    seen = set()
    for cap, line in zip(cve.CAPITALS, lines):
        pos = cap / cve.PORTFOLIO_N
        cost = (0.02 + 2 * illiq * pos / 1e6) * 100
        adv = pos / turn * 100
        best = 13.3
        want = ("DEAD" if best - cost <= 0 else "NOT EXECUTABLE" if adv > cve.ADV_CEILING_PCT
                else "SURVIVES")
        assert want in line, (cap, line)
        assert f"net {best - cost:+6.1f}%" in line
        seen.add(want)
    assert seen == {"SURVIVES", "NOT EXECUTABLE", "DEAD"}      # the data exercises all three branches
    assert "Conservative: C-S is a LOWER bound" in verdict


def test_adv_ceiling_is_applied_in_the_verdict(cve, world, capsys, monkeypatch):
    monkeypatch.setattr(cve, "ADV_CEILING_PCT", 1e9)         # nothing can exceed it
    _, out = run(cve, capsys)
    v = out.split("=== VERDICT ===")[1]
    assert "NOT EXECUTABLE" not in v and "DEAD" in v and "SURVIVES" in v
    monkeypatch.setattr(cve, "ADV_CEILING_PCT", 0.0)         # everything exceeds it
    _, out = run(cve, capsys)
    v = out.split("=== VERDICT ===")[1]
    assert "SURVIVES" not in v and "NOT EXECUTABLE (>0% of ADV)" in v


def test_adv_equal_to_the_ceiling_is_still_executable(cve, world, capsys, monkeypatch):
    am = cve.amihud(duckdb.connect()).set_index("Symbol")
    turn = am.loc[["S0", "S1", "S2"]].turnover.median()
    cap = 10_000
    pos = cap / cve.PORTFOLIO_N
    monkeypatch.setattr(cve, "ADV_CEILING_PCT", pos / turn * 100)        # exactly the position's ADV share
    _, out = run(cve, capsys)
    line = [l for l in out.split("=== VERDICT ===")[1].splitlines() if l.strip().startswith("$") and "10,000 " in l][0]
    assert "SURVIVES" in line


def test_corwin_schultz_keeps_zero_range_days_as_pair_partners(cve, tmp_path):
    flat_days = symbol_frame("Z", 0.01, 1e4, n=260)
    flat_days.loc[flat_days.index[::10], ["High", "Low"]] = 100.0        # High == Low on every 10th bar
    cve.PX = write_parquet(flat_days, tmp_path / "p.parquet")
    r = cve.corwin_schultz(duckdb.connect()).iloc[0]
    # High >= Low keeps those bars, so each still pairs with its neighbours (259 pairs, none dropped)
    assert r.n == 259
