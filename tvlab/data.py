"""데이터 로더.

1) **TradingView CSV 내보내기** (차트 우상단 ⋯ → *Export chart data*): 가격과
   화면에 켜진 모든 지표 플롯이 함께 나온다 → 지표 수식 검증용으로 최적.
2) **yfinance** (선택): ``pip install yfinance``. 한국 종목은 ``005930.KS`` 형식.
3) **synthetic**: 네트워크 없이 테스트/데모용 랜덤워크.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_tradingview_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    lower = {c: c.lower() for c in df.columns if c.lower() in {"time", "open", "high", "low", "close", "volume"}}
    df = df.rename(columns=lower)
    t = df.pop("time")
    if np.issubdtype(t.dtype, np.number):
        idx = pd.to_datetime(t, unit="s", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    df.index = idx.dt.tz_convert(None).rename("time")
    if "volume" not in df:
        df["volume"] = np.nan
    return df.sort_index()


def load_yfinance(ticker: str, start: str = "2005-01-01", interval: str = "1d") -> pd.DataFrame:
    import yfinance as yf  # optional dependency

    df = yf.download(ticker, start=start, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def synthetic(n: int = 3000, seed: int = 0, drift: float = 0.0003, vol: float = 0.015) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[100, close[:-1]] * np.exp(rng.normal(0, vol / 4, n))
    span = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + span / 2
    low = np.minimum(open_, close) - span / 2
    vol_ = rng.lognormal(13, 0.4, n)
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol_}, index=idx)
