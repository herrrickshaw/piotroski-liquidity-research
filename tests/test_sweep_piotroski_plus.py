import datetime as dt
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from helpers import bdays, walk, write_parquet

NAN = float("nan")


@pytest.fixture
def sw(load, tmp_path, monkeypatch):
    mod = load("sweep_piotroski_plus")
    (tmp_path / "reports").mkdir()
    monkeypatch.chdir(tmp_path)
    return mod


def row(**kw):
    base = dict(net_income=100.0, ta=1000.0, cfo=150.0, borrowings=100.0, shares=100.0, revenue=1000.0,
                roce=0.20, raw_material_cost=NAN, change_in_inventory=NAN, power_and_fuel=NAN,
                other_mfr_exp=NAN)
    base.update(kw)
    return SimpleNamespace(**base)


def prev(**kw):
    return row(**{**dict(net_income=60.0, cfo=100.0, borrowings=200.0, revenue=900.0), **kw})


# ── tests() ──────────────────────────────────────────────────────────────────
def test_all_available_tests_pass_and_test_six_is_always_skipped(sw):
    d = sw.tests(row(), prev(), [0.1, 0.15, 0.2])
    for n in ("1_roa_positive", "2_cfo_positive", "3_roa_improving", "4_accruals_cfo_gt_roa",
              "5_leverage_falling", "7_no_dilution", "9_asset_turnover_rising"):
        assert d[n] is True, n
    assert d["6_current_ratio_rising"] is None          # proxy failed validation: skipped, never zero
    assert d["8_gross_margin_rising"] is None           # no raw-material cost -> not a manufacturer
    assert d["10_roce_level"] is True
    assert d["11_roce_stable"] is True and d["12_roce_not_deteriorating"] is True
    assert len(d) == 12


@pytest.mark.parametrize("cur,test", [
    (dict(net_income=-1.0), "1_roa_positive"),
    (dict(cfo=-1.0), "2_cfo_positive"),
    (dict(net_income=50.0), "3_roa_improving"),
    (dict(cfo=50.0), "4_accruals_cfo_gt_roa"),
    (dict(borrowings=250.0), "5_leverage_falling"),
    (dict(shares=102.0), "7_no_dilution"),
    (dict(revenue=800.0), "9_asset_turnover_rising"),
    (dict(roce=0.15), "10_roce_level"),
])
def test_each_available_test_can_fail(sw, cur, test):
    assert sw.tests(row(**cur), prev(), [0.1, 0.15, 0.2])[test] is False


def test_dilution_tolerance_boundary(sw):
    assert sw.tests(row(shares=101.0), prev(shares=100.0), [])["7_no_dilution"] is True
    assert sw.tests(row(shares=101.5), prev(shares=100.0), [])["7_no_dilution"] is False


def test_missing_inputs_skip_instead_of_failing(sw):
    d = sw.tests(row(cfo=NAN, borrowings=NAN, shares=NAN, revenue=NAN, roce=NAN), prev(), [])
    for n in ("2_cfo_positive", "4_accruals_cfo_gt_roa", "5_leverage_falling", "7_no_dilution",
              "9_asset_turnover_rising", "10_roce_level"):
        assert d[n] is None, n
    assert d["1_roa_positive"] is True
    assert sw.tests(row(), prev(borrowings=NAN), [])["5_leverage_falling"] is None
    assert sw.tests(row(), prev(shares=0.0), [])["7_no_dilution"] is None      # zero prior share count
    assert sw.tests(row(ta=0.0), prev(), [])["1_roa_positive"] is None
    assert sw.tests(row(), prev(ta=0.0), [])["3_roa_improving"] is None


def test_roce_stability_and_trend(sw):
    d = sw.tests(row(roce=0.25), prev(), [0.25, 0.25, 0.25])
    assert d["11_roce_stable"] is True and d["12_roce_not_deteriorating"] is True     # >= mean (equal)
    d = sw.tests(row(roce=0.1), prev(), [0.1, 0.3, 0.2])
    assert d["12_roce_not_deteriorating"] is False
    d = sw.tests(row(roce=0.2), prev(), [0.01, 0.4, 0.2])
    assert d["11_roce_stable"] is False
    d = sw.tests(row(), prev(), [0.2, 0.2])                                  # < 3 years of history
    assert d["11_roce_stable"] is None and d["12_roce_not_deteriorating"] is None
    d = sw.tests(row(), prev(), [0.004, 0.004, 0.004])                       # |mean| <= 0.01
    assert d["11_roce_stable"] is None


def test_roce_thresholds_come_from_piotroski_plus(sw):
    import piotroski_plus as PP
    assert sw.ROCE_CV_HURDLE == PP.ROCE_CV_HURDLE and sw.ROCE_LEVEL_HURDLE == PP.ROCE_LEVEL_HURDLE
    assert sw.tests(row(roce=PP.ROCE_LEVEL_HURDLE + 1e-9), prev(), [])["10_roce_level"] is True


def mfr(rm, rev=1000.0, chg=0.0, pf=0.0, om=0.0, **kw):
    return dict(raw_material_cost=rm, revenue=rev, change_in_inventory=chg, power_and_fuel=pf,
                other_mfr_exp=om, **kw)


def gm_test(sw, cur_kw, prv_kw):
    return sw.tests(row(**cur_kw), prev(**prv_kw), [])["8_gross_margin_rising"]


def test_gross_margin_is_derived_only_for_manufacturers(sw):
    # cogs = rm - change_in_inventory + power + other; margin = (rev - cogs)/rev
    assert gm_test(sw, mfr(500, pf=50, om=50), mfr(600, rev=900.0, pf=50, om=50)) is True
    assert gm_test(sw, mfr(600), mfr(500)) is False
    # inventory build lowers cogs
    assert gm_test(sw, mfr(500, chg=100), mfr(500)) is True
    # not a manufacturer: raw materials < 30% of sales -> skipped, not guessed
    assert gm_test(sw, mfr(299), mfr(299)) is None
    assert gm_test(sw, mfr(300), mfr(300)) is False         # exactly 30% IS a manufacturer (equal margins: not rising)
    assert gm_test(sw, mfr(500), mfr(299)) is None          # one year ungated -> skipped
    assert gm_test(sw, mfr(500, rev=0.0), mfr(500)) is None  # zero revenue guarded
    assert gm_test(sw, mfr(NAN), mfr(500)) is None


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="sweep_piotroski_plus.py:136 — `r.change_in_inventory or 0` keeps a NaN (truthy), "
                          "so a missing change_in_inventory/power_and_fuel/other_mfr_exp makes the margin NaN "
                          "and test 8 is scored False instead of skipped")
def test_missing_cost_lines_skip_the_margin_test(sw):
    assert gm_test(sw, mfr(500, chg=NAN), mfr(500, chg=NAN)) is None


# ── main() ───────────────────────────────────────────────────────────────────
TICKERS = [f"G{i:02d}" for i in range(16)] + [f"B{i:02d}" for i in range(16)]
START = "2018-07-02"
NB = 2150
CAL = bdays(START, NB)
BASE_FY = 2015


def fund_rows(tk, direction, years=range(2016, 2027)):
    rows = []
    for fy in years:
        k = fy - 2016
        d = direction
        ni = 40 + 6 * k * d
        ce = 700.0
        ebit = 60 + 8 * k * d
        rows.append(dict(ticker=tk, fy_end=pd.Timestamp(f"{fy}-03-31"), net_income=float(ni),
                         revenue=800.0 + 50 * k * d, cfo=1.5 * ni, shares=100.0, equity_capital=100.0,
                         reserves=400.0 + 20 * k * d, borrowings=200.0 - 10 * k * d, other_liab=300.0,
                         receivables=1.0, inventory=1.0, cash=1.0, ebit=float(ebit), capital_employed=ce,
                         roce=ebit / ce, raw_material_cost=None, change_in_inventory=None,
                         power_and_fuel=None, other_mfr_exp=None))
    return rows


def prices(extra=None):
    frames = []
    for i, tk in enumerate(TICKERS):
        close = walk(NB, i + 1, 100.0, 0.012, 0.0004 if tk.startswith("G") else -0.0001)
        frames.append(pd.DataFrame({"Date": CAL, "Symbol": tk, "Close": close}))
    return frames + (extra or [])


@pytest.fixture
def world(sw, tmp_path):
    funds = []
    for tk in TICKERS:
        funds += fund_rows(tk, +1 if tk.startswith("G") else -1)
    # lag-boundary names: FY end exactly 120 days before the 2019 rebalance vs one day later
    for nm, fy2 in (("EDGE_IN", "2019-04-03"), ("EDGE_OUT", "2019-04-04")):
        base = fund_rows(nm, +1, years=[2018, 2026])
        base[0]["fy_end"] = pd.Timestamp("2018-03-31")
        base[1]["fy_end"] = pd.Timestamp(fy2)
        funds += base
    funds += fund_rows("NOPX", +1)                      # fundamentals but no prices
    px_extra = [pd.DataFrame({"Date": CAL, "Symbol": nm, "Close": walk(NB, 99, 100.0, 0.01)})
                for nm in ("EDGE_IN", "EDGE_OUT")]
    sw.FUND = write_parquet(pd.DataFrame(funds), tmp_path / "f.parquet")
    pxs = pd.concat(prices(px_extra), ignore_index=True)
    sw.PX = write_parquet(pxs, tmp_path / "px.parquet")
    return SimpleNamespace(px=pxs, funds=pd.DataFrame(funds), mod=sw, tmp=tmp_path)


def run_main(w, capsys):
    rc = w.mod.main()
    out = capsys.readouterr().out
    return rc, out, pd.read_csv(w.tmp / "reports" / "sweep_piotroski_plus_india.csv", parse_dates=["t"])


def test_load_filters_and_derives_total_assets(sw, tmp_path):
    rows = fund_rows("A", +1)[:2] + fund_rows("NOEBIT", +1)[:1] + fund_rows("NOCE", +1)[:1]
    rows[2]["ebit"] = None
    rows[3]["capital_employed"] = 0.0
    zero_ta = fund_rows("ZEROTA", +1)[:1]
    for k in ("equity_capital", "reserves", "borrowings", "other_liab"):
        zero_ta[0][k] = 0.0
    sw.FUND = write_parquet(pd.DataFrame(rows + zero_ta), tmp_path / "f.parquet")
    con, f = sw.load()
    assert set(f.ticker) == {"A"}
    a = f.iloc[0]
    assert a.ta == 100 + 400 + 200 + 300
    assert list(f.fy_end) == sorted(f.fy_end)
    # null balance-sheet pieces count as zero in the total
    r = fund_rows("N", +1)[:1]
    r[0]["reserves"] = None
    r[0]["borrowings"] = None
    sw.FUND = write_parquet(pd.DataFrame(r), tmp_path / "f2.parquet")
    assert sw.load()[1].ta.iloc[0] == 100 + 300


def test_panel_rows_cover_every_visible_ticker_and_rebalance(sw, world, capsys):
    rc, out, d = run_main(world, capsys)
    assert rc == 0
    assert set(d.columns) == {"Symbol", "t", "fwd_ret", "canonical", "quality", "turnaround", "safety"}
    regular = d[d.Symbol.isin(TICKERS)]
    assert len(regular) == len(TICKERS) * 8
    assert sorted(d.t.unique()) == [pd.Timestamp(f"{y}-08-01") for y in range(2018, 2026)]
    assert "NOPX" not in set(d.Symbol)                 # no price -> no forward return -> not scored
    assert f"scored panel: {len(d):,} stock-years | {d.Symbol.nunique():,} symbols | 8 rebalances" in out
    assert "Educational/research only. NOT investment advice." in out


def test_forward_return_is_252_bars_from_the_first_bar_on_or_after_the_rebalance(sw, world, capsys):
    _, _, d = run_main(world, capsys)
    for tk, t in (("G03", "2018-08-01"), ("B05", "2020-08-01"), ("G10", "2025-08-01")):
        close = world.px[world.px.Symbol == tk].set_index("Date").Close
        i = close.index.searchsorted(pd.Timestamp(t))
        want = close.iloc[i + 252] / close.iloc[i] - 1
        got = d[(d.Symbol == tk) & (d.t == t)].fwd_ret.iloc[0]
        assert got == pytest.approx(want)


def test_fundamentals_visible_only_if_year_end_is_at_least_120_days_old(sw, world, capsys):
    _, _, d = run_main(world, capsys)
    have = {(s, pd.Timestamp(t)) for s, t in zip(d.Symbol, d.t)}
    assert ("EDGE_IN", pd.Timestamp("2019-08-01")) in have        # 2019-04-03 == t - 120d: inclusive
    assert ("EDGE_OUT", pd.Timestamp("2019-08-01")) not in have   # one day late: only one year visible
    assert ("EDGE_OUT", pd.Timestamp("2020-08-01")) in have
    assert ("EDGE_IN", pd.Timestamp("2018-08-01")) not in have    # a single visible year is not enough
    assert sw.LAG_DAYS == 120 and sw.HOLD == 252 and sw.TOP_N == 20


def test_scores_match_an_independent_weighing_of_the_tests(sw, world, capsys):
    import piotroski_plus as PP
    _, _, d = run_main(world, capsys)
    con, f = sw.load()
    for tk, t in (("G05", "2020-08-01"), ("B07", "2023-08-01"), ("G00", "2018-08-01")):
        g = f[f.ticker == tk].reset_index(drop=True)
        vis = g[g.fy_end <= pd.Timestamp(t) - dt.timedelta(days=120)]
        r = sw.tests(vis.iloc[-1], vis.iloc[-2], vis["roce"].dropna().tolist())
        got = d[(d.Symbol == tk) & (d.t == t)].iloc[0]
        for p in PP.PRESETS:
            assert got[p] == pytest.approx(PP.weigh(r, p)["pct"])
    # improving names outscore deteriorating ones on the canonical control
    last = d[d.t == "2025-08-01"].set_index("Symbol").canonical
    assert last[[s for s in TICKERS if s.startswith("G")]].min() > last[[s for s in TICKERS if s.startswith("B")]].max()


def test_corporate_action_inside_the_window_drops_the_stock_for_that_rebalance(sw, world, tmp_path, capsys):
    px = world.px.copy()
    tk = "G02"
    c = px[px.Symbol == tk].set_index("Date").Close
    i = c.index.searchsorted(pd.Timestamp("2021-08-01"))
    px.loc[(px.Symbol == tk) & (px.Date >= c.index[i + 100]), "Close"] *= 0.3     # -70% inside the hold
    sw.PX = write_parquet(px, tmp_path / "px2.parquet")
    _, _, d = run_main(world, capsys)
    have = {(s, pd.Timestamp(t)) for s, t in zip(d.Symbol, d.t)}
    assert (tk, pd.Timestamp("2021-08-01")) not in have
    assert (tk, pd.Timestamp("2020-08-01")) in have and (tk, pd.Timestamp("2022-08-01")) in have


def parse_portfolio_rows(out):
    sec = out.split("TOP-20 PORTFOLIO PER VECTOR")[1].split("excess vs universe")[0]
    rows = {}
    for l in sec.splitlines():
        t = l.split()
        if t and t[0] in ("canonical", "quality", "turnaround", "safety", "[universe]"):
            rows[t[0]] = t[1:]
    return rows


def test_top_n_portfolio_table_matches_the_csv(sw, world, capsys):
    import piotroski_plus as PP
    _, out, d = run_main(world, capsys)
    rows = parse_portfolio_rows(out)
    uni = d.groupby("t").fwd_ret.mean()
    assert float(rows["[universe]"][0].rstrip("%")) == pytest.approx(round(uni.mean() * 100, 1), abs=0.05)
    assert float(rows["[universe]"][4].rstrip("%")) == pytest.approx(round(uni.min() * 100, 1), abs=0.05)
    for p in PP.PRESETS:
        pr = []
        for t, g in d.groupby("t"):
            g2 = g[g[p].notna()]
            pr.append((g2.nlargest(20, p).fwd_ret.mean(), g2.fwd_ret.mean()))
        port = np.array([x for x, _ in pr]); un = np.array([u for _, u in pr])
        r = rows[p]
        assert float(r[0].rstrip("%")) == pytest.approx(round(port.mean() * 100, 1), abs=0.05)
        assert float(r[1].rstrip("%")) == pytest.approx(round(np.median(port) * 100, 1), abs=0.05)
        assert float(r[2].rstrip("%")) == pytest.approx(round((port - un).mean() * 100, 1), abs=0.05)
        assert r[3] == f"{int(((port - un) > 0).sum())}/{len(pr)}"      # "beat univ" is a count of years


def test_year_by_year_table_has_a_row_per_rebalance(sw, world, capsys):
    _, out, d = run_main(world, capsys)
    sec = out.split("excess vs universe, year by year")[1]
    years = [int(l.split()[0]) for l in sec.splitlines() if l.strip()[:4].isdigit()]
    assert years == list(range(2018, 2026))
    assert "8 rebalances. canonical is the CONTROL" in sec


def test_vectors_with_fewer_than_top_n_names_are_skipped(sw, world, tmp_path, capsys):
    funds = world.funds[world.funds.ticker.isin(TICKERS[:10] + TICKERS[16:26])]
    sw.FUND = write_parquet(funds, tmp_path / "few.parquet")
    rc, out, d = run_main(world, capsys)
    assert rc == 0 and d.groupby("t").size().max() == 20         # exactly TOP_N: still enough
    assert set(parse_portfolio_rows(out)) == {"[universe]", "canonical", "quality", "turnaround", "safety"}
    sw.FUND = write_parquet(world.funds[world.funds.ticker.isin(TICKERS[:9] + TICKERS[16:26])], tmp_path / "few2.parquet")
    rc, out, d = run_main(world, capsys)
    assert d.groupby("t").size().max() == 19
    assert parse_portfolio_rows(out).keys() == {"[universe]"}   # no vector has a full 20-name pool


def test_no_scored_rows_returns_one(sw, world, tmp_path, capsys):
    sw.FUND = write_parquet(world.funds[world.funds.ticker == "NOPX"], tmp_path / "np.parquet")
    assert sw.main() == 1
    assert "no scored rows" in capsys.readouterr().out
