"""TradingView 인기 내장 지표 — 기본 파라미터를 TradingView 와 동일하게 둔다.

모든 함수는 OHLCV DataFrame(``open, high, low, close, volume`` 소문자 컬럼)을 받아
같은 인덱스의 Series/DataFrame 을 돌려준다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ta


# ── 추세 (Trend) ────────────────────────────────────────────────────────────
def ma_ribbon(df, lengths=(5, 10, 20, 60, 120, 240), kind="sma", src="close") -> pd.DataFrame:
    """국내 HTS 관례의 이동평균 묶음 (사용자 지표 plot_0~9 의 기본 켜짐 6개)."""
    s = ta.source(df, src)
    return pd.DataFrame({f"{kind}{n}": ta.MA[kind](s, n) for n in lengths})


def supertrend(df, factor=3.0, atr_period=10) -> pd.DataFrame:
    """``ta.supertrend``. direction == -1 이 상승추세 (Pine 관례)."""
    h, l, c = df["high"], df["low"], df["close"]
    a = ta.atr(h, l, c, atr_period).to_numpy()
    hl2 = ((h + l) / 2).to_numpy()
    close = c.to_numpy()
    n = len(df)
    up, lo = hl2 + factor * a, hl2 - factor * a
    st = np.full(n, np.nan)
    d = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(a[i]):
            continue
        if i > 0 and not np.isnan(a[i - 1]):
            pl, pu = lo[i - 1], up[i - 1]
            lo[i] = lo[i] if (lo[i] > pl or close[i - 1] < pl) else pl
            up[i] = up[i] if (up[i] < pu or close[i - 1] > pu) else pu
            if st[i - 1] == pu:
                d[i] = -1 if close[i] > up[i] else 1
            else:
                d[i] = 1 if close[i] < lo[i] else -1
        else:
            d[i] = 1
        st[i] = lo[i] if d[i] == -1 else up[i]
    return pd.DataFrame({"supertrend": st, "direction": d}, index=df.index)


def dmi(df, di_len=14, adx_len=14) -> pd.DataFrame:
    """``ta.dmi`` — +DI, -DI, ADX."""
    h, l, c = df["high"], df["low"], df["close"]
    up, down = ta.change(h), -ta.change(l)
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    trur = ta.rma(ta.tr(h, l, c), di_len)
    plus = 100 * ta.rma(plus_dm, di_len) / trur
    minus = 100 * ta.rma(minus_dm, di_len) / trur
    s = plus + minus
    adx = 100 * ta.rma((plus - minus).abs() / s.where(s != 0, 1), adx_len)
    return pd.DataFrame({"plus_di": plus, "minus_di": minus, "adx": adx})


def ichimoku(df, conv=9, base=26, span_b=52, displacement=26) -> pd.DataFrame:
    """일목균형표. ``cloud_a/cloud_b`` 는 **현재 봉 위에 그려진** 구름 값
    (= ``displacement-1`` 봉 전에 계산된 선행스팬) 이라 미래 참조가 없다."""
    h, l = df["high"], df["low"]
    mid = lambda n: (ta.highest(h, n) + ta.lowest(l, n)) / 2  # noqa: E731
    tenkan, kijun = mid(conv), mid(base)
    lead_a, lead_b = (tenkan + kijun) / 2, mid(span_b)
    k = displacement - 1
    return pd.DataFrame({"tenkan": tenkan, "kijun": kijun,
                         "cloud_a": lead_a.shift(k), "cloud_b": lead_b.shift(k)})


def psar(df, start=0.02, inc=0.02, maximum=0.2) -> pd.Series:
    """Parabolic SAR (Wilder). 초반 몇 봉은 TradingView 와 다를 수 있다."""
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 2:
        return pd.Series(out, index=df.index)
    up = h[1] >= h[0]
    sar, ep, af = (l[0], h[1], start) if up else (h[0], l[1], start)
    out[1] = sar
    for i in range(2, n):
        sar = sar + af * (ep - sar)
        if up:
            sar = min(sar, l[i - 1], l[i - 2])
            if l[i] < sar:
                up, sar, ep, af = False, ep, l[i], start
            elif h[i] > ep:
                ep, af = h[i], min(af + inc, maximum)
        else:
            sar = max(sar, h[i - 1], h[i - 2])
            if h[i] > sar:
                up, sar, ep, af = True, ep, h[i], start
            elif l[i] < ep:
                ep, af = l[i], min(af + inc, maximum)
        out[i] = sar
    return pd.Series(out, index=df.index)


# ── 모멘텀 (Momentum / Oscillators) ────────────────────────────────────────
def rsi(df, length=14, src="close") -> pd.Series:
    ch = ta.change(ta.source(df, src))
    up = ta.rma(ch.clip(lower=0), length)
    down = ta.rma((-ch).clip(lower=0), length)
    r = 100 - 100 / (1 + up / down)
    return r.where(down != 0, 100).where(up != 0, 0).where(up.notna() & down.notna())


def macd(df, fast=12, slow=26, signal=9, src="close") -> pd.DataFrame:
    s = ta.source(df, src)
    m = ta.ema(s, fast) - ta.ema(s, slow)
    sig = ta.ema(m, signal)
    return pd.DataFrame({"macd": m, "signal": sig, "hist": m - sig})


def stoch(df, k_len=14, k_smooth=1, d_len=3) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    hh, ll = ta.highest(h, k_len), ta.lowest(l, k_len)
    k = ta.sma(100 * (c - ll) / (hh - ll), k_smooth)
    return pd.DataFrame({"k": k, "d": ta.sma(k, d_len)})


def stoch_rsi(df, k=3, d=3, rsi_len=14, stoch_len=14) -> pd.DataFrame:
    r = rsi(df, rsi_len)
    raw = 100 * (r - ta.lowest(r, stoch_len)) / (ta.highest(r, stoch_len) - ta.lowest(r, stoch_len))
    kk = ta.sma(raw, k)
    return pd.DataFrame({"k": kk, "d": ta.sma(kk, d)})


def cci(df, length=20, src="hlc3") -> pd.Series:
    s = ta.source(df, src)
    return (s - ta.sma(s, length)) / (0.015 * ta.dev(s, length))


def williams_r(df, length=14) -> pd.Series:
    hh, ll = ta.highest(df["high"], length), ta.lowest(df["low"], length)
    return 100 * (df["close"] - hh) / (hh - ll)


# ── 변동성 (Volatility / Bands) ────────────────────────────────────────────
def bollinger(df, length=20, mult=2.0, src="close", ma="sma") -> pd.DataFrame:
    s = ta.source(df, src)
    basis = ta.MA[ma](s, length)
    dev = mult * ta.stdev(s, length)
    up, lo = basis + dev, basis - dev
    return pd.DataFrame({"basis": basis, "upper": up, "lower": lo,
                         "pct_b": (s - lo) / (up - lo), "bandwidth": (up - lo) / basis})


def donchian(df, length=20) -> pd.DataFrame:
    up, lo = ta.highest(df["high"], length), ta.lowest(df["low"], length)
    return pd.DataFrame({"upper": up, "lower": lo, "basis": (up + lo) / 2})


def keltner(df, length=20, mult=2.0, atr_len=10, use_ema=True) -> pd.DataFrame:
    c = df["close"]
    basis = ta.ema(c, length) if use_ema else ta.sma(c, length)
    rng = ta.atr(df["high"], df["low"], c, atr_len)
    return pd.DataFrame({"basis": basis, "upper": basis + mult * rng, "lower": basis - mult * rng})


def atr(df, length=14) -> pd.Series:
    return ta.atr(df["high"], df["low"], df["close"], length)


# ── 거래량 (Volume) ───────────────────────────────────────────────────────
def obv(df) -> pd.Series:
    return (np.sign(ta.change(df["close"])).fillna(0) * df["volume"]).cumsum()


def mfi(df, length=14, src="hlc3") -> pd.Series:
    s = ta.source(df, src)
    ch = ta.change(s)
    flow = df["volume"] * s
    upper = flow.where(ch > 0, 0.0).rolling(length, min_periods=length).sum()
    lower = flow.where(ch < 0, 0.0).rolling(length, min_periods=length).sum()
    return 100 - 100 / (1 + upper / lower)


def vwap_session(df) -> pd.Series:
    """일봉 이하 차트용 세션 VWAP (날짜가 바뀌면 리셋)."""
    s = ta.source(df, "hlc3")
    day = pd.Index(pd.to_datetime(df.index)).date
    pv = (s * df["volume"]).groupby(day).cumsum()
    v = df["volume"].groupby(day).cumsum()
    return pv / v
