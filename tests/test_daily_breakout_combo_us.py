import datetime as dt
import re

import numpy as np
import pandas as pd
import pytest
from helpers import bdays, write_parquet

N = 800
CAL = bdays("2018-01-01", N)
M = CAL[-1]
T_IN = M - dt.timedelta(days=282)          # last signal date the script still scores
HOLD = 252

# three fiscal-year states; S0 -> S1 improves every Piotroski test, S1 -> S2 fails all but test 4
S0 = dict(net_income=50, cfo=80, total_assets=1000, current_assets=400, current_liabilities=300,
          long_term_debt=300, shares=100, revenue=900, gross_profit=300)
S1 = dict(net_income=100, cfo=150, total_assets=1000, current_assets=500, current_liabilities=250,
          long_term_debt=200, shares=100, revenue=1000, gross_profit=400)
S2 = dict(net_income=-50, cfo=-10, total_assets=1000, current_assets=300, current_liabilities=250,
          long_term_debt=300, shares=105, revenue=900, gross_profit=200)
F_IMPROVING, F_WORSENING = 9.0, 1.0


def px(name, bk, g=0.0, vol=1000.0, second=None, split_at=None, n=N):
    """Flat 100, a first-ever close above the 60-bar box at bar `bk`, then a flat plateau."""
    c = np.full(n, 100.0)
    c[bk] = 110.0
    if second is None:
        c[bk + 1:] = 110.0 * (1 + g)
    else:
        c[bk + 1:second] = 105.0
        c[second:] = 115.0
    if split_at is not None:
        c[split_at:] *= 0.4
    return pd.DataFrame({"Date": CAL[:n], "Symbol": name, "Close": c, "High": c, "Volume": vol})


def fund(name, states, filed, fy0=2015):
    rows = []
    for k, (st, f) in enumerate(zip(states, filed)):
        fy = pd.Timestamp(f"{fy0 + k}-12-31")
        rows.append({"ticker": name, "fy_end": fy, "filed": pd.Timestamp(f), **st})
    return rows


def filings(start_fy, k):
    return [f"{start_fy + i + 1}-03-01" for i in range(k)]


class World:
    def __init__(self):
        self.px, self.fund = [], []

    def add(self, name, pxdf, fund_rows=()):
        self.px.append(pxdf)
        self.fund.extend(fund_rows)

    def good(self, name, bk, states, vol, g=0.0, **kw):
        self.add(name, px(name, bk, g, vol, **kw), fund(name, states, filings(2015, len(states))))


def standard(w, n_hi=25, n_lo=25):
    for i in range(n_hi):
        w.good(f"HI{i}", 100 + i % 5, [S0, S1], 1000 * (2 * i + 1), g=(0.0 if i % 2 else 0.05 * (i % 4 - 1)))
    for i in range(n_lo):
        w.good(f"LO{i}", 100 + i % 5, [S0, S1, S2], 1000 * (2 * i + 2), g=0.02 * (i % 3 - 1))
    return w


@pytest.fixture
def dbc(load, tmp_path, monkeypatch):
    mod = load("daily_breakout_combo_us")
    (tmp_path / "reports").mkdir()
    monkeypatch.chdir(tmp_path)

    def run(world, capsys):
        stale = tmp_path / "reports" / "daily_breakout_combo_us.csv"
        if stale.exists():
            stale.unlink()
        mod.PX = write_parquet(pd.concat(world.px, ignore_index=True), tmp_path / "px.parquet")
        mod.FUND = write_parquet(pd.DataFrame(world.fund), tmp_path / "f.parquet")
        rc = mod.main()
        out = capsys.readouterr().out
        csv = tmp_path / "reports" / "daily_breakout_combo_us.csv"
        return rc, out, (pd.read_csv(csv, parse_dates=["Date"]) if csv.exists() else None)
    return mod, run


def test_calendar_assumptions():
    assert T_IN.weekday() < 5 and T_IN in set(CAL)
    assert (CAL[CAL.get_loc(T_IN) + 1] - M).days == -281


def test_headline_counts_and_csv(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    rc, out, d = run(w, capsys)
    assert rc == 0
    assert "DARVAS x PIOTROSKI7" in out and "NOT investment advice" in out
    n_sym = 50
    # a plateau above the box re-crosses on the next bar for the g>0 names (box includes the
    # breakout bar's high by then) -> more raw hits than kept signals
    raw = int(re.search(r"raw first-cross breakouts: ([\d,]+)", out).group(1).replace(",", ""))
    assert raw > n_sym
    m = re.search(r"after 10-bar cooldown: ([\d,]+) signals, ([\d,]+) symbols", out)
    assert (int(m.group(1)), int(m.group(2))) == (n_sym, n_sym)
    assert len(d) == n_sym and d.Symbol.nunique() == n_sym
    assert f"signals WITH point-in-time fundamentals: {n_sym}" in out
    assert list(d.columns) == ["Symbol", "Date", "fwd_ret", "turnover", "F9", "yr", "wret", "tier"]


def test_signal_is_the_first_bar_that_closes_above_the_prior_box(dbc, capsys):
    mod, run = dbc
    _, _, d = run(standard(World()), capsys)
    got = d.set_index("Symbol")
    for i in range(25):
        s = got.loc[f"HI{i}"]
        assert s.Date == CAL[100 + i % 5]          # the 110 bar, never the 100-plateau bars
    want = pd.Series({f"HI{i}": (0.0 if i % 2 else 0.05 * (i % 4 - 1)) for i in range(25)}
                     | {f"LO{i}": 0.02 * (i % 3 - 1) for i in range(25)}).reindex(got.index)
    assert got.fwd_ret.tolist() == pytest.approx(want.tolist())


def test_f_score_from_the_last_two_visible_years(dbc, capsys):
    mod, run = dbc
    _, _, d = run(standard(World()), capsys)
    got = d.set_index("Symbol").F9
    assert (got[[f"HI{i}" for i in range(25)]] == F_IMPROVING).all()
    assert (got[[f"LO{i}" for i in range(25)]] == F_WORSENING).all()


def test_fundamentals_are_point_in_time(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    bk = 120
    sig_date = CAL[bk]
    # newest year (S2) is filed AFTER the signal -> invisible -> pair is still (S1, S0)
    w.add("LATE", px("LATE", bk, vol=500), fund("LATE", [S0, S1, S2], ["2016-03-01", "2017-03-01",
                                                                       sig_date + dt.timedelta(days=1)]))
    # filed on the signal date itself -> visible (<=)
    w.add("SAMEDAY", px("SAMEDAY", bk, vol=600), fund("SAMEDAY", [S0, S1, S2], ["2016-03-01", "2017-03-01", sig_date]))
    _, _, d = run(w, capsys)
    got = d.set_index("Symbol").F9
    assert got["LATE"] == F_IMPROVING
    assert got["SAMEDAY"] == F_WORSENING


def test_symbols_without_two_visible_years_are_dropped(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    w.add("NOFUND", px("NOFUND", 110, vol=700))
    w.add("ONEYR", px("ONEYR", 110, vol=700), fund("ONEYR", [S1], ["2017-03-01"]))
    _, out, d = run(w, capsys)
    assert not {"NOFUND", "ONEYR"} & set(d.Symbol)
    m = re.search(r"after 10-bar cooldown: ([\d,]+) signals, ([\d,]+) symbols", out)
    assert int(m.group(1)) == 52                      # the two are real signals, just without fundamentals


def test_fundamental_row_filters(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    late = fund("SLOWFILER", [S0, S1], ["2016-03-01", "2018-03-01"])
    late[1]["filed"] = late[1]["fy_end"] + dt.timedelta(days=401)       # > 400 days: row rejected
    w.add("SLOWFILER", px("SLOWFILER", 110, vol=700), late)
    edge = fund("EDGE400", [S0, S1], ["2016-03-01", "2018-03-01"])
    edge[1]["filed"] = edge[1]["fy_end"] + dt.timedelta(days=400)       # exactly 400: kept
    edge[0]["filed"] = edge[0]["fy_end"] + dt.timedelta(days=0)         # 0 days: kept
    w.add("EDGE400", px("EDGE400", 110, vol=700), edge)
    nulls = fund("NOGP", [S0, S1], filings(2015, 2))
    for r in nulls:
        r["gross_profit"] = None
    w.add("NOGP", px("NOGP", 110, vol=700), nulls)
    zero_cl = fund("ZEROCL", [S0, S1], filings(2015, 2))
    zero_cl[1]["current_liabilities"] = 0
    w.add("ZEROCL", px("ZEROCL", 110, vol=700), zero_cl)
    _, _, d = run(w, capsys)
    names = set(d.Symbol)
    assert "EDGE400" in names
    assert not {"SLOWFILER", "NOGP", "ZEROCL"} & names


def test_missing_long_term_debt_skips_test_five_and_rescales(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    for nm, states in (("NOLTD_HI", [S0, S1]), ("NOLTD_LO", [S0, S1, S2])):
        rows = fund(nm, states, filings(2015, len(states)))
        for r in rows:
            r["long_term_debt"] = None
        w.add(nm, px(nm, 115, vol=800), rows)
    _, _, d = run(w, capsys)
    got = d.set_index("Symbol").F9
    assert got["NOLTD_HI"] == pytest.approx(9.0)                 # 8 of 8 -> still a perfect score
    assert got["NOLTD_LO"] == pytest.approx(1 / 8 * 9)           # 1 of 8 (test 5 not counted)


def test_cooldown_is_fourteen_calendar_days(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    for nm, gap in (("GAP10", 10), ("GAP9", 9)):
        w.good(nm, 100, [S0, S1], 900, second=100 + gap)
    _, out, d = run(w, capsys)
    assert (d.Symbol == "GAP10").sum() == 2 and (d.Symbol == "GAP9").sum() == 1
    gap10 = d[d.Symbol == "GAP10"].sort_values("Date")
    assert (gap10.Date.iloc[1] - gap10.Date.iloc[0]).days == 14


def test_corporate_action_inside_the_holding_window_excludes_the_signal(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    w.good("SPLIT_IN", 110, [S0, S1], 900, split_at=110 + HOLD)          # last bar of the window
    w.good("SPLIT_OUT", 110, [S0, S1], 900, split_at=110 + HOLD + 1)     # one bar beyond it
    _, _, d = run(w, capsys)
    assert "SPLIT_IN" not in set(d.Symbol) and "SPLIT_OUT" in set(d.Symbol)


def test_signal_must_be_282_days_before_the_last_bar(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    i = CAL.get_loc(T_IN)
    w.good("CUT_IN", i, [S0, S1], 900)
    w.good("CUT_OUT", i + 1, [S0, S1], 900)
    _, _, d = run(w, capsys)
    assert "CUT_IN" in set(d.Symbol) and "CUT_OUT" not in set(d.Symbol)


def test_signals_count_boundary_fifty(dbc, capsys):
    mod, run = dbc
    rc, out, d = run(standard(World(), 25, 25), capsys)
    assert rc == 0 and len(d) == 50
    rc, out, d = run(standard(World(), 25, 24), capsys)
    assert rc == 1 and d is None
    assert "only 49 signals with fundamentals — too few" in out


def test_winsorisation_and_tiers(dbc, capsys):
    mod, run = dbc
    w = standard(World())
    w.good("BIG", 103, [S0, S1], 1_000, g=0.9)                  # a lottery ticket (+90%, under the +100% CA band)
    _, _, d = run(w, capsys)
    for yr, g in d.groupby("yr"):
        lo, hi = g.fwd_ret.quantile(.01), g.fwd_ret.quantile(.99)
        assert g.wret.tolist() == pytest.approx(g.fwd_ret.clip(lo, hi).tolist())
    assert d.loc[d.Symbol == "BIG", "wret"].iloc[0] < d.loc[d.Symbol == "BIG", "fwd_ret"].iloc[0]
    # tertiles by turnover: SMALL < MID < LARGE
    meds = d.groupby("tier").turnover.median()
    assert meds["SMALL"] < meds["MID"] < meds["LARGE"]
    assert d.tier.value_counts().min() >= 17


def test_tier_table_counts_match_the_csv(dbc, capsys):
    mod, run = dbc
    w = standard(World(), 30, 30)
    _, out, d = run(w, capsys)
    block = out.split("DAILY-SCANNED BREAKOUTS BY TIER")[1].split("per-year")[0]
    rows = {l.split()[0]: l.split() for l in block.splitlines()
            if l.split() and l.split()[0] in ("SMALL", "MID", "LARGE")}
    assert set(rows) == {"SMALL", "MID", "LARGE"}
    for t, r in rows.items():
        g = d[d.tier == t]
        assert int(r[3].replace(",", "")) == len(g)
        assert int(r[5]) == (g.F9 >= 7).sum()
        assert int(r[7]) == (g.F9 <= 3).sum()
        base, dv = float(r[1].rstrip("%")), float(r[2].rstrip("%"))
        assert base + dv == pytest.approx(g.wret.median() * 100, abs=0.15)
        # an F>=7 median is only shown with more than 15 names in the cell
        assert (r[4].lstrip("+") == "nan%") == ((g.F9 >= 7).sum() <= 15)
        assert (r[6].lstrip("+") == "nan%") == ((g.F9 <= 3).sum() <= 15)


def test_per_year_table(dbc, capsys):
    mod, run = dbc
    _, out, d = run(standard(World(), 30, 30), capsys)
    block = out.split("does the overlay ADD?")[1]
    rows = [l.split() for l in block.splitlines() if l.strip()[:4].isdigit()]
    assert [int(r[0]) for r in rows] == sorted(d.yr.unique().tolist())
    r2018 = rows[0]
    # one year, 60 names, tertile cells of 20 with 10-15 high-F names each -> cells need >=5 / >=10
    for t_i, t in enumerate(("SMALL", "MID", "LARGE")):
        g = d[(d.yr == 2018) & (d.tier == t)]
        hi = g[g.F9 >= 7]
        cell = r2018[1 + t_i]
        if len(hi) >= 5 and len(g) >= 10:
            assert float(cell.rstrip("%")) == pytest.approx((hi.wret.median() - g.wret.median()) * 100, abs=0.06)
        else:
            assert cell == "--"
