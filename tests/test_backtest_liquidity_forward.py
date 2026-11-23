import datetime as dt
import statistics

import duckdb
import numpy as np
import pandas as pd
import pytest
from helpers import bdays, write_parquet

N = 700
CAL = bdays("2022-01-03", N)
HOLD = 63
LIQ = 60


@pytest.fixture
def blf(load):
    return load("backtest_liquidity_forward")


def sym(name, close, start=0, vol=1000.0, end=None):
    close = np.asarray(close, dtype=float)
    idx = np.arange(start, N if end is None else end)
    return pd.DataFrame({"Date": CAL[idx], "Symbol": name, "Close": close[idx], "Volume": vol})


def lin(a, b):
    return a + b * np.arange(N)


def with_split(name, at, factor=0.4):
    c = lin(100, 0.5)
    c[at:] = c[at:] * factor
    return sym(name, c)


def rebal_dates(blf, tmp_path, frames, years=1):
    path = write_parquet(pd.concat(frames, ignore_index=True), tmp_path / "px.parquet")
    blf.PARQUET = path
    con = duckdb.connect()
    blf.build(con, years)
    blf.sample(con, years)
    return con


def base_frames():
    return [sym("UP", lin(100, 0.5)),
            sym("DOWN", lin(400, -0.4), vol=500.0),
            sym("FLAT", np.full(N, 100.0), vol=2000.0)]


def test_panel_columns_and_basic_arithmetic(blf, tmp_path):
    con = rebal_dates(blf, tmp_path, base_frames())
    panel = con.execute("SELECT * FROM panel ORDER BY Symbol, t").df()
    assert not panel.empty
    cal = {d: i for i, d in enumerate(CAL)}
    close = {"UP": lin(100, 0.5), "DOWN": lin(400, -0.4), "FLAT": np.full(N, 100.0)}
    vol = {"UP": 1000.0, "DOWN": 500.0, "FLAT": 2000.0}
    for r in panel.itertuples():
        i = cal[pd.Timestamp(r.t)]
        c = close[r.Symbol]
        assert r.Close == pytest.approx(c[i])
        assert r.exit_px == pytest.approx(c[i + HOLD])
        assert r.fwd_ret == pytest.approx(c[i + HOLD] / c[i] - 1)
        assert r.liq == pytest.approx(vol[r.Symbol] * c[i - LIQ + 1:i + 1].mean())
        assert r.dma50 == pytest.approx(c[i - 49:i + 1].mean())
        assert r.dma200 == pytest.approx(c[i - 199:i + 1].mean())
        assert bool(r.above200) == bool(r.dma50 > r.dma200)
        assert not r.delisted_in_window


def test_breakout_compares_against_prior_bars_only(blf, tmp_path):
    con = rebal_dates(blf, tmp_path, base_frames())
    got = dict(con.execute("SELECT Symbol, bool_and(breakout) FROM panel GROUP BY 1").fetchall())
    # rising: each close beats the previous 60 closes. If the box included the current bar this
    # could never be true. Falling / flat series never break out.
    assert got == {"UP": True, "DOWN": False, "FLAT": False}
    got = dict(con.execute("SELECT Symbol, bool_and(above200) FROM panel GROUP BY 1").fetchall())
    assert got == {"UP": True, "DOWN": False, "FLAT": False}      # equal MAs are not 'above'


def test_box_top_is_the_prior_60_bar_maximum(blf, tmp_path):
    c = np.full(N, 100.0)
    c[300] = 150.0                     # a spike
    con = rebal_dates(blf, tmp_path, [sym("S", c)])
    top = dict(con.execute("SELECT bar_i, box_top_prev FROM feat").fetchall())
    px_start = N - con.execute("SELECT count(*) FROM px").fetchone()[0]
    spike = 300 - px_start + 1                         # bar_i of the spike (1-based)
    assert top[spike] == 100.0                         # the spike bar does not see itself
    assert top[spike + 1] == 150.0                     # next bar sees it
    assert top[spike + 60] == 150.0                    # still inside the 60-bar lookback
    assert top[spike + 61] == 100.0                    # rolled off


def test_delisted_stock_exits_at_last_close_not_dropped(blf, tmp_path):
    frames = base_frames() + [sym("GONE", lin(50, 0.2), end=560)]
    con = rebal_dates(blf, tmp_path, frames)
    d = con.execute("SELECT * FROM panel WHERE Symbol='GONE' ORDER BY t").df()
    flagged = d[d.delisted_in_window]
    assert len(flagged) >= 1
    last = 50 + 0.2 * 559
    assert flagged.exit_px.tolist() == pytest.approx([last] * len(flagged))
    assert flagged.fwd_ret.tolist() == pytest.approx((last / flagged.Close - 1).tolist())
    live = d[~d.delisted_in_window]
    assert len(live) >= 1                      # earlier rebalances were normal


def test_corporate_action_defence_boundaries(blf, tmp_path):
    base = base_frames()
    con0 = rebal_dates(blf, tmp_path, base)
    rebs = [pd.Timestamp(x) for x in con0.execute("SELECT t FROM rebal ORDER BY t").df().t]
    ti = CAL.get_loc(rebs[2])                  # an interior rebalance date (calendar index)
    tj = CAL.get_loc(rebs[3])
    frames = base + [
        with_split("AHEAD63", ti + HOLD),       # split exactly 63 bars after t  -> excluded at t
        with_split("AHEAD64", ti + HOLD + 1),   # 64 bars after                  -> kept
        with_split("BEHIND60", tj - LIQ),       # split 60 bars before t         -> excluded at t
        with_split("BEHIND61", tj - LIQ - 1),   # 61 bars before                 -> kept
    ]
    con = rebal_dates(blf, tmp_path, frames)
    have = {(s, pd.Timestamp(t)) for s, t in con.execute("SELECT Symbol, t FROM panel").fetchall()}
    assert ("AHEAD63", rebs[2]) not in have
    assert ("AHEAD64", rebs[2]) in have
    assert ("BEHIND60", rebs[3]) not in have
    assert ("BEHIND61", rebs[3]) in have
    # a 60% drop is outside the [-50%, +100%] band; a clean +100% / -50% move is not flagged
    flags = dict(con.execute("SELECT Symbol, sum(is_ca) FROM ca GROUP BY 1").fetchall())
    assert flags["UP"] == 0 and flags["AHEAD63"] == 1


@pytest.mark.parametrize("factor,flagged", [(0.5, 0), (0.499, 1), (2.0, 0), (2.01, 1)])
def test_ca_move_thresholds_are_strict(blf, tmp_path, factor, flagged):
    c = np.full(N, 100.0)
    c[400:] = 100.0 * factor
    con = rebal_dates(blf, tmp_path, [sym("X", c)])
    assert con.execute("SELECT sum(is_ca) FROM ca").fetchone()[0] == flagged


def test_history_requirements_for_liquidity_and_dma(blf, tmp_path):
    con0 = rebal_dates(blf, tmp_path, base_frames())
    rebs = [pd.Timestamp(x) for x in con0.execute("SELECT t FROM rebal ORDER BY t").df().t]
    ti = CAL.get_loc(rebs[3])
    full = lin(100, 0.5)
    frames = base_frames() + [
        sym("HIST200", full, start=ti - 199),   # exactly 200 bars of history at t -> kept
        sym("HIST199", full, start=ti - 198),   # 199 bars                          -> dropped
    ]
    con = rebal_dates(blf, tmp_path, frames)
    have = {(s, pd.Timestamp(t)) for s, t in con.execute("SELECT Symbol, t FROM panel").fetchall()}
    assert ("HIST200", rebs[3]) in have
    assert ("HIST199", rebs[3]) not in have


def test_zero_volume_and_nonpositive_prices_never_enter_the_panel(blf, tmp_path):
    bad = sym("ZEROVOL", lin(100, 0.5), vol=0.0)
    neg = sym("NEGPX", np.full(N, -5.0))
    con = rebal_dates(blf, tmp_path, base_frames() + [bad, neg])
    assert "ZEROVOL" not in {r[0] for r in con.execute("SELECT Symbol FROM panel").fetchall()}
    assert "NEGPX" not in {r[0] for r in con.execute("SELECT Symbol FROM px").fetchall()}


def test_px_window_is_years_plus_one(blf, tmp_path):
    frames = base_frames()
    path = write_parquet(pd.concat(frames, ignore_index=True), tmp_path / "px.parquet")
    blf.PARQUET = path
    con = duckdb.connect()
    blf.build(con, 1)
    lo, hi = con.execute("SELECT min(Date), max(Date) FROM px").fetchone()
    assert hi == CAL[-1]
    edge = CAL[-1] - pd.DateOffset(years=2)
    lo = pd.Timestamp(lo)
    assert edge <= lo < edge + dt.timedelta(days=5)


def test_rebalance_dates_are_first_trading_day_of_each_quarter_and_stop_hold_bars_early(blf, tmp_path):
    con = rebal_dates(blf, tmp_path, base_frames())
    rebs = [pd.Timestamp(x) for x in con.execute("SELECT t FROM rebal ORDER BY t").df().t]
    assert rebs == sorted(set(rebs))
    for t in rebs[1:]:                              # first bar of its quarter
        assert CAL[CAL.get_loc(t) - 1].quarter != t.quarter or CAL[CAL.get_loc(t) - 1].year != t.year
    assert max(rebs) <= CAL[N - HOLD]               # never later than the last date with 63 bars after it
    assert blf.HOLD_BARS == 63 and blf.LIQ_WINDOW == 60 and blf.MIN_BARS_FOR_DMA == 200


def feed_panel(con, rows):
    con.execute("CREATE TABLE rebal(t TIMESTAMP)")
    con.execute("INSERT INTO rebal VALUES ('2024-01-01'), ('2024-04-01')")
    con.execute("""CREATE TABLE panel(Symbol VARCHAR, t TIMESTAMP, liq DOUBLE, Close DOUBLE,
                   dma50 DOUBLE, dma200 DOUBLE, breakout BOOLEAN, above200 BOOLEAN,
                   exit_px DOUBLE, delisted_in_window BOOLEAN, fwd_ret DOUBLE)""")
    for r in rows:
        con.execute("INSERT INTO panel VALUES (?,?,?,?,?,?,?,?,?,?,?)", r)


def test_report_quintile_statistics(blf, capsys):
    con = duckdb.connect()
    rows = []
    # 5 symbols x 2 dates; liquidity rank is the symbol number (S1 most liquid)
    rets = {"2024-01-01": [0.10, 0.08, 0.06, 0.04, 0.02], "2024-04-01": [0.20, 0.00, -0.02, 0.06, 0.02]}
    for t, rr in rets.items():
        for k, fr in enumerate(rr):
            rows.append((f"S{k+1}", t, (5 - k) * 1e7, 100.0, 1, 1, k % 2 == 0, k < 3,
                         100 * (1 + fr), k == 4 and t == "2024-04-01", fr))
    feed_panel(con, rows)
    blf.report(con)
    out = capsys.readouterr().out
    assert "panel: 10 stock-quarters | 5 symbols | 2 rebalances | 2024-01-01..2024-04-01" in out
    assert "delisted inside a holding window: 1 (10.00%)" in out
    lines = {l.split()[0]: l.split() for l in out.splitlines() if l.strip().startswith("Q")}
    q1 = [l for l in out.splitlines() if l.strip().startswith("Q1")]
    assert len(q1) == 2                                  # one row in each table
    first = q1[0].split()
    r = [0.10, 0.20]
    assert first[1] == "2"
    assert float(first[2]) == pytest.approx(round(statistics.median([5e7, 5e7]) / 1e7, 2))
    assert float(first[3]) == pytest.approx(round(statistics.mean(r) * 100, 2))
    assert float(first[4]) == pytest.approx(round(statistics.median(r) * 100, 2))
    assert float(first[5]) == pytest.approx(round(statistics.stdev(r) * 100, 1))
    assert float(first[6]) == pytest.approx(round(statistics.mean(r) / statistics.stdev(r), 3))
    # second table: S1 (Q1) breakout at both dates (k%2==0), above200 at both, never below
    second = q1[1].split()
    assert float(second[1]) == pytest.approx(15.0)       # above200_ret
    assert second[2] == "NaN"                            # nothing below the 200-DMA
    assert float(second[4]) == pytest.approx(15.0)       # breakout_ret
    assert float(second[5]) == pytest.approx(0.0)        # breakout_edge = mean(all) - mean(all)
    q4 = [l for l in out.splitlines() if l.strip().startswith("Q4")][1].split()
    # S4: above200 False (k<3 only), breakout False -> above200_ret NaN, below200 = mean(.04,.06)
    assert q4[1] == "NaN"
    assert float(q4[2]) == pytest.approx(5.0)
    assert q4[3] == "NaN"
    assert q4[4] == "NaN"


def test_main_prints_header_and_returns_zero(blf, tmp_path, capsys, monkeypatch):
    path = write_parquet(pd.concat(base_frames(), ignore_index=True), tmp_path / "px.parquet")
    blf.PARQUET = path
    monkeypatch.setattr("sys.argv", ["x", "1"])
    assert blf.main() == 0
    out = capsys.readouterr().out
    assert "LIQUIDITY -> FORWARD RETURN | India | 1y | hold 63 bars" in out
    assert "NOT investment advice" in out
    assert "forward 63-day return by LIQUIDITY quintile" in out


def test_main_default_years_is_five(blf, tmp_path, capsys, monkeypatch):
    path = write_parquet(pd.concat(base_frames(), ignore_index=True), tmp_path / "px.parquet")
    blf.PARQUET = path
    monkeypatch.setattr("sys.argv", ["x"])
    assert blf.main() == 0
    assert "India | 5y |" in capsys.readouterr().out
