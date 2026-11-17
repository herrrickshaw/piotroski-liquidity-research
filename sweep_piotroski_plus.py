#!/usr/bin/env python3
"""
sweep_piotroski_plus.py — weight-vector sweep on REAL 10-year point-in-time data.

Supersedes backtest_piotroski_plus.py, which was starved: yfinance carries ~5 annual
statements, leaving 2 usable rebalances after the reporting lag and the prior-year a
Piotroski delta needs. This reads screener.in's collected history instead —
10 fiscal years (2017-2026) — which is what makes ~8 rebalances possible.

TESTS AVAILABLE FROM THIS SOURCE — 8 of 9, and which 8 matters
--------------------------------------------------------------
  1 ROA > 0                net_income / TA          TA = eq+res+bor+other_liab
  2 CFO > 0                cfo
  3 dROA > 0
  4 accruals CFO/TA > ROA
  5 leverage falling       borrowings / TA
  6 current ratio rising   CA ~ receivables+inventory+cash ; CL ~ other_liab   APPROX
  7 no dilution            shares
  8 gross margin rising    NOT AVAILABLE — screener.in's annual P&L has no COGS line
                           (it itemises Raw Material Cost, Power and Fuel, Employee
                           Cost etc. separately, and which of those are "cost of
                           goods" is a judgement call, not a fact). Skipped rather
                           than guessed.
  9 asset turnover rising  revenue / TA
  10-12 ROCE block         ebit / capital_employed, 10y CV, trend — ALL THREE

Test 6 is an APPROXIMATION and is labelled as one: screener.in's balance sheet does
not split current from non-current, so "Other Liabilities" stands in for current
liabilities and receivables+inventory+cash for current assets. It is directionally
right and wrong in level. It is included because excluding it would drop Piotroski's
only liquidity test entirely; it is flagged here so nobody reads test 6 as exact.

NO-LOOKAHEAD
------------
screener.in exposes the fiscal PERIOD-END but not the filing date. SEBI LODR requires
audited annuals within ~60 days of FY-end; this uses 120 to be conservative. At
rebalance t only fiscal years with fy_end + 120d <= t are visible. Rebalances are
1 August, so a 31-March year-end has cleared.

WHAT SURVIVORSHIP LOOKS LIKE HERE — better, still not clean
-----------------------------------------------------------
The universe comes from ltm/IN.parquet, which retains 964 delisted symbols. But
screener.in only serves pages for LISTED companies, so a firm that delisted in 2019
still has no fundamentals and cannot be scored. The bias is smaller than the yfinance
route (10y of history means a 2019 delisting at least had scoreable years BEFORE it
died, if it was collected) but it is not zero. Results remain an upper bound.
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd

import piotroski_plus as PP
import sweep_common as SC
from data_paths import data_path

FUND = data_path("PLR_IN_FUND")
PX = data_path("PLR_IN_PX")
LAG_DAYS = 120
HOLD = 252
TOP_N = 20
ROCE_CV_HURDLE = PP.ROCE_CV_HURDLE
ROCE_LEVEL_HURDLE = PP.ROCE_LEVEL_HURDLE


def load():
    con = duckdb.connect()
    f = con.execute(f"""
        SELECT ticker, CAST(fy_end AS DATE) fy_end, net_income, revenue, cfo, shares,
               equity_capital, reserves, borrowings, other_liab,
               receivables, inventory, cash, ebit, capital_employed, roce,
               raw_material_cost, change_in_inventory, power_and_fuel, other_mfr_exp
        FROM '{FUND}' WHERE ebit IS NOT NULL AND capital_employed > 0
    """).df()
    f["ta"] = (f["equity_capital"].fillna(0) + f["reserves"].fillna(0)
               + f["borrowings"].fillna(0) + f["other_liab"].fillna(0))
    f = f[f["ta"] > 0].sort_values(["ticker", "fy_end"])
    return con, f


def _gross_margin(r):
    """Gross margin from screener.in raw-cost lines; None unless a manufacturer (rm/sales >= 0.30)."""
    if not (pd.notna(r.raw_material_cost) and r.revenue and r.revenue > 0):
        return None
    if r.raw_material_cost / r.revenue < 0.30:      # not a manufacturer
        return None
    cogs = (r.raw_material_cost - (r.change_in_inventory or 0)
            + (r.power_and_fuel or 0) + (r.other_mfr_exp or 0))
    return (r.revenue - cogs) / r.revenue


def tests(cur, prv, hist) -> dict:
    """The 8 available Piotroski tests + the 3 ROCE tests, as raw booleans."""
    d = {}
    roa = cur.net_income / cur.ta if cur.ta else None
    roa_p = prv.net_income / prv.ta if prv.ta else None
    d["1_roa_positive"] = bool(roa > 0) if roa is not None else None
    d["2_cfo_positive"] = bool(cur.cfo > 0) if pd.notna(cur.cfo) else None
    d["3_roa_improving"] = bool(roa > roa_p) if None not in (roa, roa_p) else None
    d["4_accruals_cfo_gt_roa"] = (bool(cur.cfo / cur.ta > roa)
                                  if pd.notna(cur.cfo) and cur.ta and roa is not None else None)
    lv = cur.borrowings / cur.ta if pd.notna(cur.borrowings) and cur.ta else None
    lv_p = prv.borrowings / prv.ta if pd.notna(prv.borrowings) and prv.ta else None
    d["5_leverage_falling"] = bool(lv < lv_p) if None not in (lv, lv_p) else None
    # TEST 6 IS DROPPED — the proxy FAILED validation and was never trustworthy.
    #
    # The proxy was (receivables+inventory+cash) / other_liab, because screener.in's
    # balance sheet has no current/non-current split. It was justified on the argument
    # that Piotroski test 6 only needs the SIGN of the change, so a stable level bias
    # would cancel in the delta. That argument was never tested — it was asserted.
    #
    # Measured (validate_current_ratio_proxy.py, reconstructed from yfinance since
    # screener.in blocked us; other_liab := total_assets - equity - debt, screener's
    # own definition), 237 company-years / 60 liquid Indian tickers:
    #     LEVEL: true median 1.37 vs proxy median 0.68  — off by ~2x
    #     SIGN : 62.1% agreement vs a 57.6% majority-guess baseline
    # A 4.5pp edge over guessing, on n=177. The bias is NOT stable, so it does NOT
    # cancel in the delta. The decision rule (>=80 sound / 65-80 marginal / <65 drop)
    # was fixed BEFORE the measurement precisely so this could not be rationalised
    # afterwards. 62.1% -> DROP.
    #
    # Skipped, not zeroed: weigh() removes it from both numerator and denominator, so
    # India scores on 8 tests and stays comparable to US scores on 9. Scoring it 0
    # would silently penalise every Indian company for a data gap.
    d["6_current_ratio_rising"] = None
    d["7_no_dilution"] = (bool(cur.shares <= prv.shares * 1.01)
                          if pd.notna(cur.shares) and pd.notna(prv.shares) and prv.shares else None)
    # Test 8: gross margin, computed HERE from the raw cost lines rather than read from
    # the collector's gross_margin column. Two reasons:
    #  1. The running collection loaded the collector module BEFORE the manufacturer
    #     gate was added, so its gross_margin column is the UNGATED version — wrong for
    #     services firms (it would put TCS at ~98% against a true ~42%).
    #  2. Deriving it here keeps the gate next to the test that depends on it.
    # MANUFACTURERS ONLY (rm/sales >= 0.30), where the formula is validated against
    # yfinance's true Gross Profit at -0.1pp error (n=50). Below that it is skipped, not
    # guessed: screener.in never splits Employee Cost into direct vs SG&A, so a services
    # COGS cannot be built from this source.
    gm, gm_p = _gross_margin(cur), _gross_margin(prv)
    d["8_gross_margin_rising"] = bool(gm > gm_p) if None not in (gm, gm_p) else None
    at = cur.revenue / cur.ta if pd.notna(cur.revenue) and cur.ta else None
    at_p = prv.revenue / prv.ta if pd.notna(prv.revenue) and prv.ta else None
    d["9_asset_turnover_rising"] = bool(at > at_p) if None not in (at, at_p) else None

    d["10_roce_level"] = bool(cur.roce > ROCE_LEVEL_HURDLE) if pd.notna(cur.roce) else None
    d["11_roce_stable"], d["12_roce_not_deteriorating"] = SC.roce_stability_tests(cur.roce, hist)
    return d


def main() -> int:
    con, f = load()
    print(f"\n{'='*80}\n  PIOTROSKI PLUS — WEIGHT SWEEP | India | 10y PIT | hold {HOLD}d"
          f"\n{'='*80}")
    print("  Educational/research only. NOT investment advice.")
    print("  Survivorship: screener.in serves LISTED companies only — upper bound.\n")
    print(f"  fundamentals: {len(f):,} ticker-years | {f.ticker.nunique():,} tickers | "
          f"{f.fy_end.min()} .. {f.fy_end.max()}")

    rebals = [pd.Timestamp(f"{y}-08-01") for y in range(2018, 2026)]
    # forward returns anchored to the first trading day >= t (2021-08-01 was a SUNDAY;
    # an exact-date join silently drops such rebalances)
    ret = SC.forward_returns(con, PX, rebals, HOLD)
    rmap = SC.return_lookup(ret)
    print(f"  forward returns: {len(ret):,} rows over {ret.t.nunique()} rebalances\n")

    rows = []
    by_t = {tk: g.reset_index(drop=True) for tk, g in f.groupby("ticker")}
    for t in rebals:
        cut = t - pd.Timedelta(days=LAG_DAYS)
        for tk, g in by_t.items():
            vis = g[g.fy_end <= cut]   # cut is a Timestamp; .date() raises on datetime64
            if len(vis) < 2:
                continue
            cur, prv = vis.iloc[-1], vis.iloc[-2]
            hist = vis["roce"].dropna().tolist()
            r = tests(cur, prv, hist)
            fr = rmap.get((tk, t))
            if fr is None or fr != fr:
                continue
            rows.append({"Symbol": tk, "t": t, "fwd_ret": fr, **SC.score_vectors(r)})
    d = pd.DataFrame(rows)
    if d.empty:
        print("  no scored rows"); return 1
    print(f"  scored panel: {len(d):,} stock-years | {d.Symbol.nunique():,} symbols | "
          f"{d.t.nunique()} rebalances\n")

    res = SC.top_n_by_vector(d, TOP_N)
    SC.print_top_n_table(d, res, TOP_N)
    SC.print_excess_by_year(res, "excess vs universe, year by year — where a 'winner' is made or broken")

    n = max(len(x) for x in res.values()) if res else 0
    print(f"\n  {n} rebalances. canonical is the CONTROL: a weighted vector that cannot")
    print("  beat all-weights-1.0 is fitting noise. With single-digit observations and")
    print("  stocks co-moving within a year, effective n is nearer the rebalance count")
    print("  than the row count — read the year-by-year, not the mean.")
    d.to_csv("reports/sweep_piotroski_plus_india.csv", index=False)
    print("  -> reports/sweep_piotroski_plus_india.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
