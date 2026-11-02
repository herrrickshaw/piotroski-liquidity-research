"""Pieces shared by the India and US Piotroski-Plus weight sweeps.

The two sweep scripts differ in where fundamentals come from and how the
as-of date is derived; everything below was previously copy-pasted in both.
Pure functions take data in and return data out; the only printing helpers are
the two report tables, which keep the original byte-for-byte output format.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import piotroski_plus as PP

# 1-day moves outside [-50%, +100%] are treated as corporate actions (splits stored
# as raw close) and excluded from any holding window that contains one.
CA_DOWN, CA_UP = -0.50, 1.00


def roce_stability_tests(roce: float, hist: Sequence[float]
                         ) -> Tuple[Optional[bool], Optional[bool]]:
    """Tests 11 and 12 from the current ROCE and its per-year history (fractions).

    Returns (stable, not_deteriorating); both None when fewer than 3 years or a
    near-zero mean make the coefficient of variation meaningless.
    """
    if len(hist) >= 3 and abs(np.mean(hist)) > 0.01:
        stable = bool(np.std(hist) / abs(np.mean(hist)) < PP.ROCE_CV_HURDLE)
        return stable, bool(roce >= np.mean(hist))
    return None, None


def forward_returns(con, px_path: str, rebalances: Sequence[pd.Timestamp],
                    hold: int) -> pd.DataFrame:
    """Forward `hold`-day return per (Symbol, rebalance) from a Date/Symbol/Close parquet.

    Anchored to the first trading day >= each rebalance date (a Sunday rebalance
    would be silently dropped by an exact-date join). Delisted names exit at their
    last close; windows containing a corporate-action-like move are excluded.
    Returns columns [t, Symbol, fwd_ret].
    """
    dates = ", ".join(f"DATE '{d.date()}'" for d in rebalances)
    con.execute(f"""
    CREATE OR REPLACE TABLE px AS SELECT Date, Symbol, Close FROM '{px_path}' WHERE Close>0;
    CREATE OR REPLACE TABLE ca AS
      SELECT Symbol, Date, CASE WHEN prev>0 AND (Close/prev-1 < {CA_DOWN} OR Close/prev-1 > {CA_UP})
                                THEN 1 ELSE 0 END is_ca
      FROM (SELECT Symbol, Date, Close, lag(Close) OVER (PARTITION BY Symbol ORDER BY Date) prev FROM px);
    CREATE OR REPLACE TABLE fw AS
      SELECT p.Symbol, p.Date, p.Close,
             lead(p.Close,{hold}) OVER (PARTITION BY p.Symbol ORDER BY p.Date) fwd,
             last_value(p.Close) OVER (PARTITION BY p.Symbol ORDER BY p.Date
               ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) last_px,
             sum(c.is_ca) OVER (PARTITION BY p.Symbol ORDER BY p.Date
               ROWS BETWEEN 1 FOLLOWING AND {hold} FOLLOWING) ca_ahead
      FROM px p JOIN ca c ON p.Symbol=c.Symbol AND p.Date=c.Date;
    """)
    ret = con.execute(f"""
      WITH rb AS (SELECT unnest([{dates}]) t),
      anchor AS (SELECT rb.t, min(fw.Date) d FROM rb JOIN fw ON fw.Date >= rb.t GROUP BY rb.t)
      SELECT a.t, f.Symbol, coalesce(f.fwd, f.last_px)/f.Close - 1 fwd_ret
      FROM anchor a JOIN fw f ON f.Date = a.d
      WHERE coalesce(f.ca_ahead,0)=0 AND f.Close>0
    """).df()
    ret["t"] = pd.to_datetime(ret["t"])
    return ret


def return_lookup(ret: pd.DataFrame) -> dict:
    """{(Symbol, rebalance): fwd_ret} for O(1) joins inside the rebalance loop."""
    return {(r.Symbol, r.t): r.fwd_ret for r in ret.itertuples()}


def score_vectors(r: dict) -> dict:
    """pct under every preset for one scored company (column-per-preset record)."""
    return {p: PP.weigh(r, p)["pct"] for p in PP.PRESETS}


def top_n_by_vector(d: pd.DataFrame, top_n: int) -> dict:
    """Per preset: frame of [t, port, uni, ex] — equal-weight top-N vs universe mean."""
    res = {}
    for p in PP.PRESETS:
        pr = []
        for t, g in d.groupby("t"):
            g2 = g[g[p].notna()]
            if len(g2) < top_n:
                continue
            pr.append((t, g2.nlargest(top_n, p)["fwd_ret"].mean(), g2["fwd_ret"].mean()))
        if not pr:
            continue
        x = pd.DataFrame(pr, columns=["t", "port", "uni"])
        x["ex"] = x.port - x.uni
        res[p] = x
    return res


def print_top_n_table(d: pd.DataFrame, res: dict, hold_label_n: int) -> None:
    """The '[universe] + one row per vector' table."""
    print(f"  === TOP-{hold_label_n} PORTFOLIO PER VECTOR (12m hold, equal weight) ===")
    print(f"  {'vector':11s} {'mean':>7s} {'median':>7s} {'vs univ':>8s} {'beat univ':>10s} {'worst':>7s}")
    uni = d.groupby("t")["fwd_ret"].mean()
    print(f"  {'[universe]':11s} {uni.mean()*100:>6.1f}% {uni.median()*100:>6.1f}% "
          f"{'--':>8s} {'--':>10s} {uni.min()*100:>6.1f}%")
    for p, x in res.items():
        print(f"  {p:11s} {x.port.mean()*100:>6.1f}% {x.port.median()*100:>6.1f}% "
              f"{x.ex.mean()*100:>+7.1f}% {int((x.ex>0).sum()):>4}/{len(x):<5} {x.port.min()*100:>6.1f}%")


def print_excess_by_year(res: dict, title: str) -> None:
    """Excess return vs universe, one row per rebalance year, one column per vector."""
    print(f"\n  === {title} ===")
    print(f"  {'year':6s} " + " ".join(f"{p:>11s}" for p in res))
    for t in sorted({t for p in res for t in res[p].t}):
        cells = []
        for p in res:
            v = res[p][res[p].t == t]["ex"]
            cells.append(f"{v.iloc[0]*100:>+10.1f}%" if len(v) else f"{'--':>11s}")
        print(f"  {t.year:<6} " + " ".join(cells))
