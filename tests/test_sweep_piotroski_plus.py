"""Tests for sweep_piotroski_plus.tests (India, screener.in fields)."""
import types

import numpy as np
import pytest

import sweep_piotroski_plus as S


def _row(**kw):
    base = dict(net_income=20.0, ta=200.0, cfo=30.0, borrowings=20.0, shares=100.0,
                revenue=200.0, roce=0.2, raw_material_cost=100.0, change_in_inventory=0.0,
                power_and_fuel=10.0, other_mfr_exp=0.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_all_pass_and_test6_dropped():
    cur = _row()
    prv = _row(net_income=10.0, cfo=15.0, borrowings=40.0, revenue=150.0,
               raw_material_cost=90.0)
    d = S.tests(cur, prv, [0.2, 0.19, 0.21])
    assert d["6_current_ratio_rising"] is None
    for k in ("1_roa_positive", "2_cfo_positive", "3_roa_improving", "4_accruals_cfo_gt_roa",
              "5_leverage_falling", "7_no_dilution", "8_gross_margin_rising",
              "9_asset_turnover_rising", "10_roce_level", "11_roce_stable",
              "12_roce_not_deteriorating"):
        assert d[k] is True, k


def test_gross_margin_skipped_for_non_manufacturer():
    cur = _row(raw_material_cost=10.0)        # rm/sales 5% < 30% -> services firm
    d = S.tests(cur, _row(), [])
    assert d["8_gross_margin_rising"] is None
    assert d["11_roce_stable"] is None        # < 3 years of history


def test_missing_net_income_is_skipped_not_failed():
    # screener.in rows are only required to have ebit and capital_employed (load()), so
    # net_income can be NaN. The module's own rule: a missing input is SKIPPED, never scored 0.
    cur = _row(net_income=float("nan"))
    d = S.tests(cur, _row(net_income=10.0), [0.2, 0.2, 0.2])
    assert d["1_roa_positive"] is None
    assert d["3_roa_improving"] is None
    assert d["4_accruals_cfo_gt_roa"] is None


def test_missing_prior_net_income_skips_roa_improving():
    d = S.tests(_row(), _row(net_income=float("nan")), [])
    assert d["1_roa_positive"] is True
    assert d["3_roa_improving"] is None
