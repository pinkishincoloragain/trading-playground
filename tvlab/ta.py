"""Pine Script `ta.*` primitives, reproduced bar-for-bar.

TradingView 값과 소수점까지 맞추려면 내장 함수의 세부 규칙을 그대로 따라야 한다.
자주 틀리는 지점:

* ``ta.stdev`` 는 기본값이 **모표준편차**(N으로 나눔, ``biased=true``)다. pandas 의
  ``rolling().std()`` 는 N-1 로 나누므로 그대로 쓰면 볼린저 밴드 폭이 어긋난다.
* ``ta.ema`` / ``ta.rma`` 는 첫 값을 **SMA 로 시드**한다. pandas ``ewm(adjust=False)``
  는 첫 종가로 시드하므로 초반 수백 봉이 미세하게 다르다.
* ``ta.rma`` (Wilder) 의 alpha 는 ``1/length``, ``ta.ema`` 는 ``2/(length+1)``.
* ``ta.pivotlow/pivothigh`` 는 ``right`` 봉이 지나야 확정된다. 확정 시점 기준으로
  써야 미래 참조(look-ahead)가 없다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _s(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(x, dtype=float)


def sma(src: pd.Series, length: int) -> pd.Series:
    return _s(src).rolling(length, min_periods=length).mean()


def wma(src: pd.Series, length: int) -> pd.Series:
    w = np.arange(1, length + 1, dtype=float)
    return _s(src).rolling(length, min_periods=length).apply(lambda a: np.dot(a, w) / w.sum(), raw=True)


def _seeded_ewm(src: pd.Series, length: int, alpha: float) -> pd.Series:
    """Pine 방식 EMA: 유효값 ``length`` 개의 SMA 로 시드한 뒤 재귀."""
    src = _s(src)
    x = src.to_numpy(dtype=float)
    out = np.full_like(x, np.nan)
    seed = sma(src, length).to_numpy()
    prev = np.nan
    for i in range(len(x)):
        if np.isnan(prev):
            prev = seed[i]
        elif not np.isnan(x[i]):
            prev = alpha * x[i] + (1 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=src.index)


def ema(src: pd.Series, length: int) -> pd.Series:
    return _seeded_ewm(src, length, 2.0 / (length + 1))


def rma(src: pd.Series, length: int) -> pd.Series:
    return _seeded_ewm(src, length, 1.0 / length)


def stdev(src: pd.Series, length: int, biased: bool = True) -> pd.Series:
    return _s(src).rolling(length, min_periods=length).std(ddof=0 if biased else 1)


def dev(src: pd.Series, length: int) -> pd.Series:
    """``ta.dev``: 평균 절대 편차 (CCI 용)."""
    return _s(src).rolling(length, min_periods=length).apply(lambda a: np.abs(a - a.mean()).mean(), raw=True)


def highest(src: pd.Series, length: int) -> pd.Series:
    return _s(src).rolling(length, min_periods=length).max()


def lowest(src: pd.Series, length: int) -> pd.Series:
    return _s(src).rolling(length, min_periods=length).min()


def change(src: pd.Series, length: int = 1) -> pd.Series:
    return _s(src).diff(length)


def tr(high: pd.Series, low: pd.Series, close: pd.Series, handle_na: bool = True) -> pd.Series:
    pc = close.shift(1)
    out = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1, skipna=False)
    if handle_na:
        out = out.fillna(high - low)
    return out


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    return rma(tr(high, low, close, handle_na=True), length)


def crossover(a, b) -> pd.Series:
    a, b = _s(a), (b if isinstance(b, pd.Series) else pd.Series(b, index=_s(a).index))
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a, b) -> pd.Series:
    a, b = _s(a), (b if isinstance(b, pd.Series) else pd.Series(b, index=_s(a).index))
    return (a < b) & (a.shift(1) >= b.shift(1))


def _pivot(src: pd.Series, left: int, right: int, is_low: bool) -> pd.Series:
    """피벗 값을 **확정된 봉**(피벗 봉 + right)에 기록한다. Pine 과 동일한 배치."""
    x = _s(src).to_numpy(dtype=float)
    out = np.full_like(x, np.nan)
    for i in range(left + right, len(x)):
        c = i - right
        win = x[c - left : i + 1]
        if np.isnan(win).any():
            continue
        v = x[c]
        # Pine: 왼쪽은 엄격 부등호, 오른쪽은 같음 허용
        lhs, rhs = win[:left], win[left + 1 :]
        if is_low and (lhs > v).all() and (rhs >= v).all():
            out[i] = v
        elif not is_low and (lhs < v).all() and (rhs <= v).all():
            out[i] = v
    return pd.Series(out, index=_s(src).index)


def pivotlow(src: pd.Series, left: int, right: int) -> pd.Series:
    return _pivot(src, left, right, is_low=True)


def pivothigh(src: pd.Series, left: int, right: int) -> pd.Series:
    return _pivot(src, left, right, is_low=False)


def source(df: pd.DataFrame, name: str) -> pd.Series:
    """Pine 의 ``close / hl2 / hlc3 / ohlc4 / hlcc4`` 소스."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    return {
        "open": o, "high": h, "low": l, "close": c,
        "hl2": (h + l) / 2,
        "hlc3": (h + l + c) / 3,
        "ohlc4": (o + h + l + c) / 4,
        "hlcc4": (h + l + 2 * c) / 4,
    }[name]


MA = {"sma": sma, "ema": ema, "rma": rma, "wma": wma}
