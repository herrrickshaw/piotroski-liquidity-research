import numpy as np
import pandas as pd
import pytest

import golden_cross as GC


def frame(closes):
    return pd.DataFrame({"Close": closes})


def stepped(n_flat=250, extra=0, hi=200.0, lo=100.0):
    """n_flat bars at `lo`, then 1 + extra bars at `hi`. The 50-DMA first exceeds the 200-DMA
    on the first `hi` bar: 50-DMA 102.0 vs 200-DMA 100.5."""
    return frame([lo] * n_flat + [hi] * (1 + extra))


NULL_KEYS = {"gc_signal", "dma50_above_200", "dma50", "dma200", "dma_gap_%",
             "days_since_cross", "trend"}


def test_null_result_shape():
    r = GC.compute(None)
    assert set(r) == NULL_KEYS
    assert r == {"gc_signal": False, "dma50_above_200": False, "dma50": None, "dma200": None,
                 "dma_gap_%": None, "days_since_cross": None, "trend": "INSUFFICIENT_HISTORY"}


@pytest.mark.parametrize("bad", [None, [1, 2, 3], {"Close": [1]}, pd.DataFrame({"Open": [1.0] * 300})])
def test_bad_inputs_never_raise(bad):
    assert GC.compute(bad)["trend"] == "INSUFFICIENT_HISTORY"


def test_min_bars_boundary():
    assert GC.MIN_BARS == 201
    assert GC.compute(frame([100.0] * 200))["trend"] == "INSUFFICIENT_HISTORY"
    assert GC.compute(frame([100.0] * 201))["trend"] != "INSUFFICIENT_HISTORY"


def test_non_numeric_and_nan_closes_are_dropped_before_counting():
    closes = ["x", None] + [100.0] * 200
    assert GC.compute(frame(closes))["trend"] == "INSUFFICIENT_HISTORY"   # only 200 usable
    assert GC.compute(frame(closes + [100.0]))["trend"] == "BELOW_200DMA"  # 201 usable, flat


def test_flat_series_is_not_above():
    r = GC.compute(frame([100.0] * 300))
    assert r["trend"] == "BELOW_200DMA"        # equal MAs are NOT above
    assert r["dma50_above_200"] is False and r["gc_signal"] is False
    assert r["dma_gap_%"] == 0.0 and r["days_since_cross"] is None


def test_zero_200dma_returns_null():
    assert GC.compute(frame([0.0] * 300))["trend"] == "INSUFFICIENT_HISTORY"


def test_cross_on_last_bar():
    r = GC.compute(stepped())
    assert r["gc_signal"] is True and r["trend"] == "GOLDEN_CROSS"
    assert r["dma50_above_200"] is True
    assert r["dma50"] == 102.0 and r["dma200"] == 100.5
    assert r["dma_gap_%"] == round((102.0 - 100.5) / 100.5 * 100, 2) == 1.49
    assert r["days_since_cross"] == 0


def test_days_since_cross_counts_bars_after_the_cross():
    r = GC.compute(stepped(extra=5))
    assert r["gc_signal"] is False and r["trend"] == "ABOVE_200DMA"
    assert r["dma50_above_200"] is True
    assert r["days_since_cross"] == 5


def test_lookback_boundary():
    # The window is the last lookback+1 bars; a cross is a transition between two bars in it, so
    # the oldest detectable cross is lookback-1 bars ago. Pinned as-is (documented in the PR).
    assert GC.compute(stepped(extra=59))["days_since_cross"] == 59
    assert GC.compute(stepped(extra=60))["days_since_cross"] is None
    assert GC.compute(stepped(extra=60), lookback=61)["days_since_cross"] == 60
    assert GC.compute(stepped(extra=3), lookback=4)["days_since_cross"] == 3
    assert GC.compute(stepped(extra=4), lookback=4)["days_since_cross"] is None


def test_values_are_rounded_to_two_places():
    closes = list(np.linspace(100, 160, 260))
    r = GC.compute(frame(closes))
    s = pd.Series(closes)
    assert r["dma50"] == round(s.rolling(50).mean().iloc[-1], 2)
    assert r["dma200"] == round(s.rolling(200).mean().iloc[-1], 2)


def test_downtrend_is_below_with_negative_gap():
    r = GC.compute(frame(list(np.linspace(200, 100, 300))))
    assert r["trend"] == "BELOW_200DMA" and r["dma50_above_200"] is False
    assert r["dma_gap_%"] < 0 and r["days_since_cross"] is None


def test_most_recent_cross_wins():
    # up-cross, fall back below, up-cross again: days_since measures the LATER one
    closes = [100.0] * 250 + [200.0] * 80 + [20.0] * 90 + [400.0] * 40
    r = GC.compute(frame(closes))
    s = pd.Series(closes)
    above = (s.rolling(50).mean() > s.rolling(200).mean())
    last_cross = max(i for i in range(1, len(s)) if above[i] and not above[i - 1])
    assert r["days_since_cross"] == len(s) - 1 - last_cross
    assert r["days_since_cross"] == 26


def test_row_fields_maps_names_and_handles_null():
    r = GC.row_fields(stepped(extra=3))
    assert r == {"DMA50": 108.0, "DMA200": 102.0, "DMA_Gap%": 5.88, "GC_Signal": "ABOVE_200DMA",
                 "DMA50_above_200": True, "Days_Since_Cross": 3}
    n = GC.row_fields(None)
    assert n["GC_Signal"] == "INSUFFICIENT_HISTORY" and n["DMA50"] is None
    assert n["DMA50_above_200"] is False and n["Days_Since_Cross"] is None
