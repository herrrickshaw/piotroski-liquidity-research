"""Locations of the (untracked) parquet panels, overridable by environment variable.

Defaults keep the original machine-specific paths so behaviour is unchanged; set the
variable to point a checkout at its own copy of the panel.
"""
from __future__ import annotations

import os

_REPOS = "/Users/umashankar/repos"
_DEFAULTS = {
    "PLR_US_FUND": f"{_REPOS}/global-stock-screener/cache_seed/fundamentals_history/US.parquet",
    "PLR_IN_FUND": f"{_REPOS}/global-stock-screener/cache_seed/fundamentals_history/IN.parquet",
    "PLR_US_PX": f"{_REPOS}/global-stock-screener/cache_seed/ltm/US.parquet",
    "PLR_IN_PX": f"{_REPOS}/global-market-data/cache_seed/ltm/IN.parquet",
}


def data_path(name: str) -> str:
    """Panel path for `name` (PLR_US_FUND, PLR_IN_FUND, PLR_US_PX, PLR_IN_PX): env var, else default."""
    return os.environ.get(name, _DEFAULTS[name])
