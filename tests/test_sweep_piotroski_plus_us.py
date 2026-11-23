import datetime as dt
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from helpers import bdays, walk, write_parquet

START = "2017-05-01"
NB = 2450
CAL = bdays(START, NB)
N_PER_TIER = 22
REBAL = [pd.Timestamp(f"{y}-06-01") for y in range(2017, 2026)]


def tickers():
    out = []
    for i in range(3 * N_PER_TIER):
        out.append(f"{'G' if i % 2 == 0 else 'B'}{i:02d}")
    return out


TICKERS = tickers()


def fund_rows(tk, direction, fy_range=range(2014, 2026), filed_lag=60):
    rows = []
    for fy in fy_range:
        k = fy - 2014
        d = direction
        ni = 60 + 6 * k * d
        ta = 1000.0 + 10 * k * d
        rev = 800.0 + 50 * k * d
        rows.append(dict(
            ticker=tk, fy_end=pd.Timestamp(f"{fy}-12-31"),
            filed=pd.Timestamp(f"{fy}-12-31") + dt.timedelta(days=filed_lag),
            ebit=100.0 + 10 * k * d, net_income=float(ni), cfo=1.5 * ni, total_assets=ta,
            current_assets=400.0 + 20 * k * d, current_liabilities=300.0,
            long_term_debt=200.0 - 10 * k * d, shares=100.0, revenue=rev,
            gross_profit=rev * (0.30 + 0.01 * k * d), equity=500.0))
    return rows


def expected_booleans(cur, prv, hist):
    """Independent oracle: the 12 tests exactly as the module docstrings define them."""
    roa, roa_p = cur["net_income"] / cur["total_assets"], prv["net_income"] / prv["total_assets"]
    roce = cur["ebit"] / (cur["total_assets"] - cur["current_liabilities"])
    out = {
        "1_roa_positive": roa > 0, "2_cfo_positive": cur["cfo"] > 0, "3_roa_improving": roa > roa_p,
        "4_accruals_cfo_gt_roa": cur["cfo"] / cur["total_assets"] > roa,
        "5_leverage_falling": cur["long_term_debt"] / cur["total_assets"] < prv["long_term_debt"] / prv["total_assets"],
        "6_current_ratio_rising": cur["current_assets"] / cur["current_liabilities"] > prv["current_assets"] / prv["current_liabilities"],
        "7_no_dilution": cur["shares"] <= prv["shares"] * 1.01,
        "8_gross_margin_rising": cur["gross_profit"] / cur["revenue"] > prv["gross_profit"] / prv["revenue"],
        "9_asset_turnover_rising": cur["revenue"] / cur["total_assets"] > prv["revenue"] / prv["total_assets"],
        "10_roce_level": roce > 0.15,
    }
    if len(hist) >= 3 and abs(np.mean(hist)) > 0.01:
        out["11_roce_stable"] = np.std(hist) / abs(np.mean(hist)) < 0.30
        out["12_roce_not_deteriorating"] = roce >= np.mean(hist)
    else:
        out["11_roce_stable"] = out["12_roce_not_deteriorating"] = None
    return out


@pytest.fixture
def world(load, tmp_path, monkeypatch):
    mod = load("sweep_piotroski_plus_us")
    (tmp_path / "reports").mkdir()
    monkeypatch.chdir(tmp_path)
    funds, px = [], []
    for i, tk in enumerate(TICKERS):
        funds += fund_rows(tk, +1 if tk.startswith("G") else -1)
        close = walk(NB, i + 1, 50.0, 0.012, 0.0004 if tk.startswith("G") else 0.0)
        vol = 10_000.0 * (1 + i) ** 2            # distinct, widely spread turnover
        px.append(pd.DataFrame({"Date": CAL, "Symbol": tk, "Close": close, "Volume": vol}))
    for nm, lag in (("EDGE_IN", 152), ("EDGE_OUT", 153)):          # FY2020 filed on / one day after 2021-06-01
        rows = fund_rows(nm, +1)
        for r in rows:
            if r["fy_end"].year == 2020:
                r["filed"] = pd.Timestamp("2021-06-01") + dt.timedelta(days=lag - 152)
                # FY2020 is a bad year, so seeing it or not changes the score
                r.update(net_income=-50.0, cfo=-10.0, long_term_debt=400.0, current_assets=300.0,
                         shares=105.0, gross_profit=100.0, revenue=700.0)
        funds += rows
        px.append(pd.DataFrame({"Date": CAL, "Symbol": nm, "Close": walk(NB, 500, 50.0, 0.01),
                                "Volume": 5e9}))
    mod.FUND = write_parquet(pd.DataFrame(funds), tmp_path / "f.parquet")
    pxs = pd.concat(px, ignore_index=True)
    mod.PX = write_parquet(pxs, tmp_path / "px.parquet")
    return SimpleNamespace(mod=mod, funds=pd.DataFrame(funds), px=pxs, tmp=tmp_path)


def run(w, capsys):
    rc = w.mod.main()
    out = capsys.readouterr().out
    return rc, out, pd.read_csv(w.tmp / "reports" / "sweep_piotroski_plus_us.csv", parse_dates=["t"])


def test_constants(world):
    m = world.mod
    assert (m.HOLD, m.TOP_N) == (252, 20) and len(m.REBAL) == 9
    assert m.REBAL[0] == pd.Timestamp("2017-06-01") and m.REBAL[-1] == pd.Timestamp("2025-06-01")


def test_panel_shape_and_headline(world, capsys):
    rc, out, d = run(world, capsys)
    assert rc == 0
    assert set(d.columns) == {"Symbol", "t", "fwd_ret", "canonical", "quality", "turnaround", "safety"}
    reg = d[d.Symbol.isin(TICKERS)]
    # FY(t-1) is filed Mar 1 of year t; two years visible from t=2016 filings on -> every rebalance scores
    assert len(reg) == len(TICKERS) * len(REBAL)
    assert "PIOTROSKI PLUS — WEIGHT SWEEP | US (SEC EDGAR) | hold 252d" in out
    assert "Entry gated on the REAL SEC filing date" in out
    n = len(world.funds)
    assert f"fundamentals: {n:,} ticker-years | {world.funds.ticker.nunique():,} tickers" in out
    assert f"scored panel: {len(d):,} stock-years | {d.Symbol.nunique():,} symbols | 9 rebalances" in out


def test_scores_match_the_independent_oracle(world, capsys):
    import piotroski_plus as PP
    _, _, d = run(world, capsys)
    f = world.funds
    for tk, t in (("G00", "2019-06-01"), ("B01", "2022-06-01"), ("G10", "2025-06-01"), ("B21", "2017-06-01")):
        g = f[f.ticker == tk].sort_values("fy_end").reset_index(drop=True)
        vis = g[g.filed <= pd.Timestamp(t)]
        cur, prv = vis.iloc[-1], vis.iloc[-2]
        hist = ((vis.ebit) / (vis.total_assets - vis.current_liabilities)).tolist()
        want = expected_booleans(cur, prv, hist)
        got = d[(d.Symbol == tk) & (d.t == t)].iloc[0]
        for p in PP.PRESETS:
            assert got[p] == pytest.approx(PP.weigh(want, p)["pct"]), (tk, t, p)
    # the improving group sits above the deteriorating group at every rebalance on the control vector
    for t, g in d[d.Symbol.isin(TICKERS)].groupby("t"):
        up = g[g.Symbol.str.startswith("G")].canonical
        dn = g[g.Symbol.str.startswith("B")].canonical
        assert up.median() > dn.median()


def test_entry_is_gated_on_the_filing_date_inclusive(world, capsys):
    _, _, d = run(world, capsys)
    f = world.funds
    jun21 = pd.Timestamp("2021-06-01")
    by = lambda tk: d[(d.Symbol == tk) & (d.t == jun21)].iloc[0]
    import piotroski_plus as PP
    for tk, expect_year in (("EDGE_IN", 2020), ("EDGE_OUT", 2019)):       # newest VISIBLE fiscal year
        g = f[f.ticker == tk].sort_values("fy_end").reset_index(drop=True)
        vis = g[g.filed <= jun21]
        assert vis.iloc[-1].fy_end.year == expect_year
        want = expected_booleans(vis.iloc[-1], vis.iloc[-2], (vis.ebit / (vis.total_assets - vis.current_liabilities)).tolist())
        assert by(tk).canonical == pytest.approx(PP.weigh(want, "canonical")["pct"])
    assert by("EDGE_IN").canonical < by("EDGE_OUT").canonical            # the bad year is seen only on the filing date


def test_forward_return_and_corporate_action_filter(world, tmp_path, capsys):
    rc, out, d = run(world, capsys)
    tk, t = "G04", pd.Timestamp("2020-06-01")
    close = world.px[world.px.Symbol == tk].set_index("Date").Close
    i = close.index.searchsorted(t)
    assert d[(d.Symbol == tk) & (d.t == t)].fwd_ret.iloc[0] == pytest.approx(close.iloc[i + 252] / close.iloc[i] - 1)
    # a -70% print inside the holding window removes that stock-year (and only that one)
    px = world.px.copy()
    px.loc[(px.Symbol == tk) & (px.Date >= close.index[i + 50]), "Close"] *= 0.3
    world.mod.PX = write_parquet(px, tmp_path / "px2.parquet")
    _, _, d2 = run(world, capsys)
    have = {(s, pd.Timestamp(x)) for s, x in zip(d2.Symbol, d2.t)}
    assert (tk, t) not in have
    assert (tk, pd.Timestamp("2019-06-01")) in have and (tk, pd.Timestamp("2021-06-01")) in have


def test_filters_on_fundamentals_rows(world, tmp_path, capsys):
    funds = world.funds.copy()
    keep = funds[funds.ticker.isin(TICKERS)]
    bad = []
    base = fund_rows("BADROWS", +1)
    for col in ("net_income", "cfo", "ebit", "current_assets"):
        b = [dict(r) for r in fund_rows(f"NULL_{col}", +1)]
        for r in b:
            r[col] = None
        bad += b
    for col, val in (("total_assets", 0), ("current_liabilities", 0), ("shares", 0), ("revenue", 0)):
        b = [dict(r) for r in fund_rows(f"ZERO_{col}", +1)]
        for r in b:
            r[col] = val
        bad += b
    slow = [dict(r) for r in fund_rows("SLOW", +1)]
    for r in slow:
        r["filed"] = r["fy_end"] + dt.timedelta(days=401)
    old = [dict(r) for r in fund_rows("TOOOLD", +1, fy_range=range(2010, 2014))]
    ok_gp = [dict(r) for r in fund_rows("NO_GP", +1)]
    for r in ok_gp:
        r["gross_profit"] = None
    ok_ltd = [dict(r) for r in fund_rows("NO_LTD", +1)]
    for r in ok_ltd:
        r["long_term_debt"] = None
    allf = pd.concat([keep, pd.DataFrame(bad + slow + old + ok_gp + ok_ltd)], ignore_index=True)
    world.mod.FUND = write_parquet(allf, tmp_path / "f2.parquet")
    pxs = world.px
    extra = [pd.DataFrame({"Date": CAL, "Symbol": s, "Close": walk(NB, 7, 50.0, 0.01), "Volume": 1e6})
             for s in set(allf.ticker) - set(pxs.Symbol)]
    world.mod.PX = write_parquet(pd.concat([pxs] + extra, ignore_index=True), tmp_path / "px3.parquet")
    _, out, d = run(world, capsys)
    names = set(d.Symbol)
    # gross_profit and long_term_debt are NOT required: those tests are skipped, the company stays
    assert {"NO_GP", "NO_LTD"} <= names
    assert not any(n.startswith(("NULL_", "ZERO_")) for n in names)
    assert not names & {"SLOW", "TOOOLD"}
    gp = d[d.Symbol == "NO_GP"].set_index("t").canonical
    assert gp.notna().all() and (gp.max() <= 100)


def test_missing_gross_profit_rescales_instead_of_penalising(world, tmp_path, capsys):
    import piotroski_plus as PP
    funds = world.funds.copy()
    keep = funds[funds.ticker.isin(TICKERS)]
    gp = pd.DataFrame(fund_rows("NO_GP", +1))
    gp["gross_profit"] = None
    full = pd.DataFrame(fund_rows("WITH_GP", +1))
    allf = pd.concat([keep, gp, full], ignore_index=True)
    world.mod.FUND = write_parquet(allf, tmp_path / "f2.parquet")
    extra = [pd.DataFrame({"Date": CAL, "Symbol": s, "Close": walk(NB, 3, 50.0, 0.01), "Volume": 1e6})
             for s in ("NO_GP", "WITH_GP")]
    world.mod.PX = write_parquet(pd.concat([world.px] + extra, ignore_index=True), tmp_path / "px2.parquet")
    _, _, d = run(world, capsys)
    a = d[(d.Symbol == "NO_GP") & (d.t == "2025-06-01")].iloc[0]
    b = d[(d.Symbol == "WITH_GP") & (d.t == "2025-06-01")].iloc[0]
    # an improving company that passes everything scores 100 with or without the gross-margin line
    assert a.canonical == pytest.approx(100.0) and b.canonical == pytest.approx(100.0)


def section(out, start, end=None):
    s = out.split(start)[1]
    return s.split(end)[0] if end else s


def test_portfolio_and_year_tables(world, capsys):
    import piotroski_plus as PP
    _, out, d = run(world, capsys)
    sec = section(out, "TOP-20 PORTFOLIO PER VECTOR", "excess vs universe")
    rows = {l.split()[0]: l.split()[1:] for l in sec.splitlines() if l.split() and l.split()[0] in
            ("canonical", "quality", "turnaround", "safety", "[universe]")}
    assert set(rows) == {"[universe]", *PP.PRESETS}
    uni = d.groupby("t").fwd_ret.mean()
    assert float(rows["[universe]"][0].rstrip("%")) == pytest.approx(round(uni.mean() * 100, 1), abs=0.05)
    for p in PP.PRESETS:
        pr = [(g[g[p].notna()].nlargest(20, p).fwd_ret.mean(), g[g[p].notna()].fwd_ret.mean())
              for _, g in d.groupby("t")]
        port = np.array([a for a, _ in pr]); un = np.array([b for _, b in pr])
        assert float(rows[p][0].rstrip("%")) == pytest.approx(round(port.mean() * 100, 1), abs=0.05)
        assert float(rows[p][2].rstrip("%")) == pytest.approx(round((port - un).mean() * 100, 1), abs=0.05)
        assert rows[p][3] == f"{int(((port - un) > 0).sum())}/9"
    yrs = section(out, "year by year (read THIS, not the mean)", "is high-F actually GOOD")
    assert [int(l.split()[0]) for l in yrs.splitlines() if l.strip()[:4].isdigit()] == list(range(2017, 2026))


def test_f_band_table(world, capsys):
    _, out, d = run(world, capsys)
    sec = section(out, "is high-F actually GOOD in the US?", "did this actually reach")
    for lo, hi, lab in ((0, 40, "weak"), (40, 70, "middle"), (70, 101, "strong")):
        s = d[(d.canonical >= lo) & (d.canonical < hi)]
        line = [l for l in sec.splitlines() if lab in l]
        if len(s) > 50:
            assert f"n={len(s):>5,}" in line[0]
            assert f"mean fwd {s.fwd_ret.mean() * 100:>6.1f}%" in line[0]
            assert f"median {s.fwd_ret.median() * 100:>6.1f}%" in line[0]
        else:
            assert line == []


def liq_table(world):
    px = world.px.sort_values(["Symbol", "Date"])
    last = px.groupby("Symbol").tail(60)
    return (last.assign(tv=last.Close * last.Volume).groupby("Symbol").tv.median())


def test_reach_section_reports_liquidity_percentile(world, capsys):
    _, out, d = run(world, capsys)
    liq = liq_table(world)
    tested = d.Symbol.unique()
    med = liq[tested].median()
    sec = section(out, "did this actually reach the small names?", "F-score edge BY LIQUIDITY")
    assert f"tested symbols {len(tested):,} | median turnover ${med / 1e6:,.1f}M/day" in sec
    assert f"full universe  {len(liq):,} | median turnover ${liq.median() / 1e6:,.1f}M/day" in sec
    pct = (liq < med).mean() * 100
    assert f"sits at the {pct:.0f}th percentile" in sec


def test_liquidity_tier_edge_table(world, capsys):
    _, out, d = run(world, capsys)
    liq = liq_table(world).rename("turnover").reset_index()
    dl = d.merge(liq, on="Symbol", how="left")
    dl["tier"] = pd.qcut(dl.turnover, 3, labels=["SMALL", "MID", "LARGE"])
    sec = section(out, "F-score edge BY LIQUIDITY", "RANK WITHIN TIER")
    for t in ("SMALL", "MID", "LARGE"):
        s = dl[dl.tier == t]
        hi, lo = s[s.canonical >= 70], s[s.canonical < 40]
        line = [l for l in sec.splitlines() if l.strip().startswith(t)]
        assert len(hi) > 20 and len(lo) > 20 and len(line) == 1
        tok = line[0].split()
        assert int(tok[1].replace(",", "")) == len(s)
        assert float(tok[2].rstrip("%")) == pytest.approx(round(hi.fwd_ret.mean() * 100, 1), abs=0.05)
        assert float(tok[-1].rstrip("%")) == pytest.approx(round((hi.fwd_ret.median() - lo.fwd_ret.median()) * 100, 1), abs=0.05)


def test_rank_within_tier_table(world, capsys):
    import piotroski_plus as PP
    _, out, d = run(world, capsys)
    liq = liq_table(world).rename("turnover").reset_index()
    dl = d.merge(liq, on="Symbol", how="left")
    dl["tier"] = pd.qcut(dl.turnover, 3, labels=["SMALL", "MID", "LARGE"])
    sec = section(out, "RANK WITHIN TIER")
    for t in ("SMALL", "MID", "LARGE"):
        st = dl[dl.tier == t]
        for p in PP.PRESETS:
            per = []
            for _, g in st.groupby("t"):
                g2 = g[g[p].notna()]
                if len(g2) < 20:
                    continue
                top = g2.nlargest(10, p)
                per.append((top.fwd_ret.median(), g2.fwd_ret.median(), top.fwd_ret.mean()))
            line = [l for l in sec.splitlines() if l.split()[:2] == [t, p]]
            if len(per) < 4:
                assert line == []
                continue
            x = np.array(per)
            tok = line[0].split()
            assert float(tok[4].rstrip("%")) == pytest.approx(round((x[:, 0] - x[:, 1]).mean() * 100, 1), abs=0.05)
            assert tok[-1] == f"{int((x[:, 0] > x[:, 1]).sum())}/{len(per)}"
    assert "-> reports/sweep_piotroski_plus_us.csv" in out


@pytest.mark.xfail(strict=True, raises=AttributeError,
                   reason="sweep_piotroski_plus_us.py:271 — the tier column is only created when the merged panel "
                          "has > 200 rows (:239-240) but the RANK WITHIN TIER loop is not guarded, so a small panel "
                          "dies with AttributeError: 'DataFrame' object has no attribute 'tier'")
def test_small_panel_skips_the_tier_sections_instead_of_crashing(world, tmp_path, capsys):
    funds = world.funds[world.funds.ticker.isin(TICKERS[:12])]
    world.mod.FUND = write_parquet(funds, tmp_path / "small.parquet")
    rc, out, d = run(world, capsys)
    assert rc == 0 and len(d) <= 200


def test_no_scored_rows_returns_one(world, tmp_path, capsys):
    world.mod.FUND = write_parquet(world.funds[world.funds.ticker.isin(["EDGE_OUT"]) & (world.funds.fy_end.dt.year < 2015)],
                                   tmp_path / "none.parquet")
    assert world.mod.main() == 1
    assert "no scored rows" in capsys.readouterr().out


def test_pool_size_edges_exactly_twenty_names(world, tmp_path, capsys):
    # Only the 20 lowest-turnover names have two visible fiscal years at the first rebalance, so the
    # first-year pool is exactly TOP_N overall and exactly 2 x TOP_T inside the SMALL tier.
    funds = world.funds
    liq = liq_table(world)
    lowest = set(liq[TICKERS].sort_values().index[:20])         # turnover = close x volume, not just volume
    late = ~funds.ticker.isin(lowest) & (funds.fy_end.dt.year < 2016)
    keep = ~late & ~funds.ticker.str.startswith("EDGE")
    world.mod.FUND = write_parquet(funds[keep], tmp_path / "late.parquet")
    _, out, d = run(world, capsys)
    assert (d.t == "2017-06-01").sum() == 20
    port = section(out, "TOP-20 PORTFOLIO PER VECTOR", "excess vs universe")
    for l in port.splitlines():
        if l.split() and l.split()[0] in ("canonical", "quality", "turnaround", "safety"):
            assert l.split()[4].endswith("/9")                                   # 2017's pool of exactly 20 counts
    tiers = section(out, "RANK WITHIN TIER")
    small = [l.split() for l in tiers.splitlines() if l.split()[:1] == ["SMALL"]]
    mid = [l.split() for l in tiers.splitlines() if l.split()[:1] == ["MID"]]
    assert small and mid
    assert all(r[-1].endswith("/9") for r in small)          # 2017's pool of exactly 20 is ranked
    assert all(r[-1].endswith("/8") for r in mid)            # MID has no 2017 pool
