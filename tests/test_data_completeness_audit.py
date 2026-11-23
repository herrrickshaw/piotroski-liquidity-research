import pandas as pd
import pytest
from helpers import write_parquet


def test_reconciliation_counts_distinct_tickers_and_overlap(load, tmp_path, capsys):
    m = load("data_completeness_audit")
    # fundamentals: 5 tickers (A repeated across years); prices: 8 symbols; overlap A, B, C
    fund = pd.DataFrame({"ticker": ["A", "A", "A", "B", "C", "D", "E"], "x": range(7)})
    px = pd.DataFrame({"Symbol": ["A", "B", "C", "F", "G", "H", "I", "J", "A"], "y": range(9)})
    m.FUND = write_parquet(fund, tmp_path / "f.parquet")
    m.PX = write_parquet(px, tmp_path / "p.parquet")
    assert m.main() == 0
    out = capsys.readouterr().out
    assert "=== source reconciliation ===" in out
    assert "fundamentals      5 | prices      8 | overlap      3  (60% / 38%)" in out


def test_disjoint_sources(load, tmp_path, capsys):
    m = load("data_completeness_audit")
    m.FUND = write_parquet(pd.DataFrame({"ticker": ["A", "B"]}), tmp_path / "f.parquet")
    m.PX = write_parquet(pd.DataFrame({"Symbol": ["X", "Y", "Z", "W"]}), tmp_path / "p.parquet")
    assert m.main() == 0
    assert "fundamentals      2 | prices      4 | overlap      0  (0% / 0%)" in capsys.readouterr().out


def test_paths_point_at_the_us_panels(load):
    m = load("data_completeness_audit")
    assert m.FUND.endswith("fundamentals_history/US.parquet") and m.PX.endswith("ltm/US.parquet")
