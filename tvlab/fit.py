"""TradingView 에서 내보낸 플롯 값으로 지표 수식을 역추적한다.

상단/하단 밴드를 각각 맞추면 오차가 남지만(예: 0.92% / 2.31%), 밴드를
**중심 = (상단+하단)/2**, **반폭 = (상단-하단)/2** 로 분해하면 중심은 이동평균,
반폭은 k×(표준편차|ATR|범위) 하나로 떨어진다. 이 모듈은 그 탐색을 자동화한다.

    >>> from tvlab import fit
    >>> fit.fit_band(df, upper=df["plot_11"], lower=df["plot_12"]).head()
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ta

SOURCES = ("close", "hl2", "hlc3", "ohlc4", "open", "high", "low")


def _err(target: pd.Series, cand: pd.Series) -> float:
    m = target.notna() & cand.notna()
    if m.sum() < 10:
        return np.inf
    return float(((cand[m] - target[m]).abs() / target[m].abs().clip(lower=1e-12)).mean())


def fit_line(df: pd.DataFrame, target: pd.Series, lengths=range(2, 301),
             sources=SOURCES, kinds=("sma", "ema", "rma"), top=5) -> pd.DataFrame:
    """한 줄짜리 플롯(이동평균 등)에 맞는 ``kind(source, length)`` 를 찾는다.

    평균 상대오차(MAPE) 오름차순. wma 는 느리므로 필요할 때만 kinds 에 "wma" 를 추가.
    """
    rows = []
    for src in sources:
        s = ta.source(df, src)
        for kind in kinds:
            f = ta.MA[kind]
            for n in lengths:
                rows.append((kind, src, n, _err(target, f(s, n))))
    return pd.DataFrame(rows, columns=["kind", "src", "length", "mape"]).nsmallest(top, "mape")


def _width_candidates(df: pd.DataFrame, src: pd.Series, n: int):
    yield "stdev", ta.stdev(src, n)
    yield "stdev_sample", ta.stdev(src, n, biased=False)
    yield "atr", ta.atr(df["high"], df["low"], df["close"], n)
    yield "range", (ta.highest(df["high"], n) - ta.lowest(df["low"], n)) / 2


def fit_band(df: pd.DataFrame, upper: pd.Series, lower: pd.Series,
             lengths=range(2, 301), sources=SOURCES, kinds=("sma", "ema", "rma"), top=5) -> pd.DataFrame:
    """밴드형 지표(볼린저·켈트너·돈치안·엔벨로프)를 중심/반폭으로 분해해 맞춘다.

    반폭의 배수 k 는 최소제곱으로 닫힌형태로 구한다: k = <w, t> / <w, w>.
    """
    center = (upper + lower) / 2
    half = (upper - lower) / 2
    c_fit = fit_line(df, center, lengths, sources, kinds, top=top)
    rows = []
    for _, r in c_fit.iterrows():
        s = ta.source(df, r.src)
        for n in lengths:
            for wname, w in _width_candidates(df, s, n):
                m = half.notna() & w.notna() & (w != 0)
                if m.sum() < 10:
                    continue
                k = float((w[m] * half[m]).sum() / (w[m] ** 2).sum())
                rows.append({"center": f"{r.kind}({r.src},{r.length})", "center_mape": r.mape,
                             "width": f"{k:.4g}*{wname}({r.src if wname.startswith('stdev') else 'hlc'},{n})",
                             "k": k, "width_mape": _err(half, k * w)})
    return pd.DataFrame(rows).nsmallest(top, ["width_mape"]).reset_index(drop=True)


def is_donchian(df: pd.DataFrame, upper: pd.Series, lengths=range(2, 301)) -> tuple[int, float] | None:
    """상단이 정확히 n봉 최고가인지 확인. 상단 값이 그 구간 고가보다 크면 즉시 탈락."""
    best = None
    for n in lengths:
        e = _err(upper, ta.highest(df["high"], n))
        if best is None or e < best[1]:
            best = (n, e)
    return best
