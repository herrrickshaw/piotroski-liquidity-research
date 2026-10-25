"""Tests for golden_cross.compute."""
import numpy as np
import pandas as pd
import pytest

import golden_cross as G


def _df(closes):
    return pd.DataFrame({"Close": closes})


def test_insufficient_history_and_bad_input():
    assert G.compute(None)["trend"] == "INSUFFICIENT_HISTORY"
    assert G.compute(_df(np.ones(200)))["trend"] == "INSUFFICIENT_HISTORY"
    assert G.compute(pd.DataFrame({"x": [1]}))["trend"] == "INSUFFICIENT_HISTORY"


def test_cross_today_detected():
    # flat at 100 for 250 bars, then fall, then a sharp rise so the 50DMA crosses the 200DMA
    closes = np.r_[np.full(150, 100.0), np.full(100, 90.0)]
    out = None
    for extra in range(1, 200):
        c = np.r_[closes, np.full(extra, 130.0)]
        out = G.compute(_df(c))
        if out["gc_signal"]:
            break
    assert out["gc_signal"] and out["trend"] == "GOLDEN_CROSS" and out["days_since_cross"] == 0
    nxt = G.compute(_df(np.r_[closes, np.full(extra + 3, 130.0)]))
    assert nxt["days_since_cross"] == 3 and nxt["trend"] == "ABOVE_200DMA"


def test_gap_hand_computed():
    c = np.r_[np.full(150, 100.0), np.full(50, 110.0), np.full(1, 110.0)]
    out = G.compute(_df(c))
    d50 = 110.0
    d200 = (149 * 100 + 51 * 110) / 200          # last 200 of 150x100 + 51x110
    assert out["dma50"] == pytest.approx(round(d50, 2))
    assert out["dma200"] == pytest.approx(round(d200, 2))
    assert out["dma_gap_%"] == pytest.approx(round((d50 - d200) / d200 * 100, 2))


def test_no_phantom_cross_when_200dma_first_becomes_available():
    # Steady uptrend, 230 bars: the 50DMA has been above the 200DMA from the first bar the
    # 200DMA exists. No crossing was ever observed, so days_since_cross must be None
    # (not "31 days ago" counted from the NaN warm-up).
    out = G.compute(_df(np.linspace(100, 200, 230)))
    assert out["dma50_above_200"] is True
    assert out["days_since_cross"] is None
