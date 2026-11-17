"""Run every characterised code path against the modules found on sys.path and
return a JSON-serialisable dict. Used to build tests/golden/*.json from the
ORIGINAL code (git worktree of main) and to assert the refactor reproduces them.
"""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import tempfile

import numpy as np
import pandas as pd

from tests import synth


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if o != o else round(float(o), 10)
    return o if o is None or isinstance(o, str) else str(o)


def capture_piotroski():
    PP = importlib.import_module("piotroski_plus")
    out = {"presets": PP.PRESETS, "tests": PP.TEST_NAMES}
    for seed in range(8):
        drop = [(), ("Gross Profit",), ("Long Term Debt",), ("Cash And Cash Equivalents",)][seed % 4]
        r = PP.score(synth.fake_ticker(seed, drop))
        out[f"score_{seed}"] = r
        out[f"verdict_{seed}"] = PP.verdict(r)
        out[f"weigh_{seed}"] = {p: PP.weigh(r, p) for p in PP.PRESETS}
        out[f"sweep_{seed}"] = PP.sweep(r)
    out["score_empty"] = PP.score(synth.FakeTicker(None, None, None))
    out["weigh_empty"] = PP.weigh({})
    csvish = {"1_roa_positive": "True", "2_cfo_positive": float("nan"), "3_roa_improving": "0"}
    out["weigh_csv"] = PP.weigh(csvish, {"1_roa_positive": 2.0, "2_cfo_positive": 1.0, "3_roa_improving": 1.0})
    return out


def capture_misc():
    GC = importlib.import_module("golden_cross")
    AL = importlib.import_module("adaptive_liquidity")
    rng = np.random.default_rng(3)
    closes = 100 + np.cumsum(rng.normal(0.1, 1.0, 400))
    out = {"gc": GC.compute(pd.DataFrame({"Close": closes})),
           "gc_short": GC.compute(pd.DataFrame({"Close": closes[:50]})),
           "gc_none": GC.compute(None), "gc_row": GC.row_fields(pd.DataFrame({"Close": closes}))}
    adv = pd.DataFrame({"adv_usd": np.r_[0, 5e3, np.geomspace(2e4, 5e7, 60)]})
    gate = AL.Gate("IN", 1_000_000, 10)
    out["classify"] = AL.classify(adv, gate).to_dict("list")
    out["capacity"] = AL.capacity(adv).to_dict("list")
    out["floors"] = [AL.scan_floor(m) for m in ("IN", "US", "", "INDIA")]
    tdf = pd.DataFrame({"Median_Turnover": np.r_[np.linspace(1, 100, 40), [np.nan]],
                        "Liquidity_Tier": ["X"] * 40 + ["UNKNOWN"]})
    out["retier"] = AL.retier(tdf).to_dict("list")
    out["describe"] = AL.describe(adv, "US")
    return out


def run_sweep(modname, panels):
    """Run a sweep script's main() on synthetic parquet; return stdout + CSV text."""
    mod = importlib.import_module(modname)
    with tempfile.TemporaryDirectory() as tmp:
        fund, px = panels(tmp)
        mod.FUND, mod.PX = fund, px
        os.makedirs(os.path.join(tmp, "reports"))
        cwd = os.getcwd()
        os.chdir(tmp)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = mod.main()
        finally:
            os.chdir(cwd)
        csvs = {}
        for f in sorted(os.listdir(os.path.join(tmp, "reports"))):
            with open(os.path.join(tmp, "reports", f)) as fh:
                csvs[f] = fh.read()
    # the temp dir path never appears in output; strip trailing whitespace only
    return {"rc": rc, "stdout": buf.getvalue(), "csv": csvs}


def capture_all():
    return {
        "piotroski": _jsonable(capture_piotroski()),
        "misc": _jsonable(capture_misc()),
        "sweep_us": run_sweep("sweep_piotroski_plus_us", synth.us_panels),
        "sweep_in": run_sweep("sweep_piotroski_plus", synth.in_panels),
    }
