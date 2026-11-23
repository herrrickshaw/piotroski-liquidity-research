"""Synthetic-data helpers. Everything is generated in memory and written under tmp_path."""
import duckdb
import numpy as np
import pandas as pd


def write_parquet(df, path):
    con = duckdb.connect()
    con.register("d", df)
    con.execute(f"COPY d TO '{path}' (FORMAT PARQUET)")
    con.close()
    return str(path)


def bdays(start, n):
    return pd.bdate_range(start, periods=n)


def walk(n, seed, start_px=100.0, vol=0.01, drift=0.0003):
    rng = np.random.RandomState(seed)
    return start_px * np.exp(np.cumsum(rng.normal(drift, vol, n)))


def px_frame(specs, start="2022-01-03", n=600, with_hl=True):
    """specs: {symbol: dict(seed, vol, base_volume, px0, n)} -> long OHLCV frame."""
    frames = []
    for sym, s in specs.items():
        nn = s.get("n", n)
        close = walk(nn, s.get("seed", 1), s.get("px0", 100.0), s.get("vol", 0.015))
        rng = np.random.RandomState(s.get("seed", 1) + 1000)
        df = pd.DataFrame({
            "Date": bdays(s.get("start", start), nn),
            "Symbol": sym,
            "Close": close,
            "Volume": s.get("base_volume", 1_000_000) * (1 + rng.rand(nn)),
        })
        if with_hl:
            df["High"] = close * (1 + rng.rand(nn) * 0.02)
            df["Low"] = close * (1 - rng.rand(nn) * 0.02)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def stmt(rows, years):
    """Statement frame like yfinance: index = line items, columns = period ends (newest first)."""
    return pd.DataFrame(rows, index=pd.to_datetime(years)).T


class FakeTicker:
    def __init__(self, income_stmt=None, balance_sheet=None, cashflow=None, boom=False):
        self._boom = boom
        self._v = (income_stmt, balance_sheet, cashflow)

    @property
    def income_stmt(self):
        if self._boom:
            raise RuntimeError("network down")
        return self._v[0]

    @property
    def balance_sheet(self):
        if self._boom:
            raise RuntimeError("network down")
        return self._v[1]

    @property
    def cashflow(self):
        if self._boom:
            raise RuntimeError("network down")
        return self._v[2]


YEARS = ["2025-03-31", "2024-03-31", "2023-03-31", "2022-03-31"]


def perfect_ticker(**over):
    """Four fiscal years in which every one of the 12 tests passes. `over` replaces whole
    line items: over={'Net Income': [..]}; a value of None deletes the line."""
    inc = {"EBIT": [200, 150, 120, 100], "Net Income": [100, 60, 50, 40],
           "Total Revenue": [1000, 900, 800, 700], "Gross Profit": [400, 300, 250, 200]}
    bal = {"Total Assets": [1000] * 4, "Current Liabilities": [200] * 4,
           "Current Assets": [500, 400, 400, 400], "Long Term Debt": [100, 200, 200, 200],
           "Ordinary Shares Number": [100, 100, 100, 100],
           "Cash And Cash Equivalents": [100, 100, 100, 100]}
    cfs = {"Operating Cash Flow": [150, 90, 80, 70]}
    for k, v in over.items():
        key = k.replace("__", " ")
        for d in (inc, bal, cfs):
            if key in d:
                if v is None:
                    del d[key]
                else:
                    d[key] = v
                break
        else:
            raise KeyError(k)
    return FakeTicker(stmt(inc, YEARS), stmt(bal, YEARS), stmt(cfs, YEARS))


def install_fake_yfinance(monkeypatch, tickers):
    """Install a fake `yfinance` module whose Ticker(sym) returns tickers[sym] (or raises)."""
    import sys
    import types

    mod = types.ModuleType("yfinance")

    def Ticker(sym):
        t = tickers.get(sym)
        if t is None:
            raise KeyError(sym)
        return t

    mod.Ticker = Ticker
    monkeypatch.setitem(sys.modules, "yfinance", mod)
    return mod


def flat_ticker(ebit=160.0, ta=1000.0, cl=200.0, cash=0.0, years=4, ebits=None, **over):
    """Constant statements: ROCE = ebit / (ta - cl) in every year (or per-year `ebits`)."""
    eb = list(ebits) if ebits is not None else [ebit] * years
    n = len(eb)
    inc = {"EBIT": eb, "Net Income": [80.0] * n, "Total Revenue": [1000.0] * n}
    bal = {"Total Assets": [ta] * n, "Current Liabilities": [cl] * n,
           "Current Assets": [500.0] * n, "Long Term Debt": [100.0] * n,
           "Cash And Cash Equivalents": [cash] * n}
    cfs = {"Operating Cash Flow": [120.0] * n}
    for k, v in over.items():
        key = k.replace("__", " ")
        for d in (inc, bal, cfs):
            if key in d:
                if v is None:
                    del d[key]
                else:
                    d[key] = v
    ys = YEARS[:n] if n <= 4 else YEARS
    return FakeTicker(stmt(inc, ys), stmt(bal, ys), stmt(cfs, ys))


PANDAS3 = int(pd.__version__.split(".")[0]) >= 3
