"""Data pipeline: download OHLCV, cache, compute indicators, chronological split.

Test period is quarantined — never used to fit normalization stats or
select hyperparameters. Only the train period informs anything the agent sees.
"""
from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache")
TRAIN_END = "2022-12-31"  # train: start .. TRAIN_END  |  test: TRAIN_END+1 .. end
STALE_DAYS = 5  # re-fetch if cached data ends more than this many days ago


def _cache_path(ticker: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker}.csv")


def _today() -> str:
    return dt.date.today().isoformat()


def download(ticker: str = "SPY", start: str = "2010-01-01", end: str | None = None,
             force: bool = False) -> pd.DataFrame:
    """Download daily OHLCV via yfinance through `end` (default today), cache to CSV.

    Auto-refreshes a stale cache so results always reflect recent data.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    end = end or _today()
    path = _cache_path(ticker)
    if os.path.exists(path) and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        last = df.index.max()
        stale = pd.isna(last) or (pd.Timestamp(end) - last).days > STALE_DAYS
        if not stale and len(df):
            return df  # cache fresh enough

    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        raise ValueError(f"yfinance returned no data for '{ticker}'. Check the symbol.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.to_csv(path)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a small, defensible indicator set. Keeps obs space tight."""
    out = df.copy()
    close = out["Close"]
    out["ret1"] = close.pct_change()
    out["sma10"] = close.rolling(10).mean()
    out["sma30"] = close.rolling(30).mean()
    out["sma_ratio"] = out["sma10"] / out["sma30"] - 1.0
    out["vol10"] = out["ret1"].rolling(10).std()
    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi"] = (100 - 100 / (1 + rs)) / 100.0  # normalized 0..1
    return out.dropna()


def split(df: pd.DataFrame, train_end: str = TRAIN_END):
    """Chronological split. No shuffle — test stays in the future."""
    train = df.loc[:train_end].copy()
    test = df.loc[train_end:].iloc[1:].copy()
    return train, test


# Feature columns the env exposes each step (plus position flag, added in env).
FEATURE_COLS = ["ret1", "sma_ratio", "vol10", "rsi"]


def load(ticker: str = "SPY", force: bool = False):
    df = add_indicators(download(ticker, force=force))
    return split(df)


if __name__ == "__main__":
    tr, te = load()
    print(f"train: {tr.index.min().date()}..{tr.index.max().date()}  n={len(tr)}")
    print(f"test : {te.index.min().date()}..{te.index.max().date()}  n={len(te)}")
    print("features:", FEATURE_COLS)
