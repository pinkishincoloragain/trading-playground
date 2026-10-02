"""매수/매도 신호 카탈로그.

규칙: 신호는 **t 봉 종가에서 확정**되고 진입은 **t+1 봉 시가**에서 한다
(``evaluate.py`` 가 그렇게 계산한다). 신호 정의 안에서 미래 봉을 보면 안 된다.

각 항목은 ``(분류, 출처/학파, 함수)``. 함수는 df -> bool Series.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from . import indicators as ind
from . import ta

Signal = Callable[[pd.DataFrame], pd.Series]


def _rsi_cross_up(level):
    return lambda df: ta.crossover(ind.rsi(df), level)


def _bb_lower_reentry(df):
    b = ind.bollinger(df)
    return ta.crossover(df["close"], b["lower"])


def _bb_squeeze_breakout(df):
    b = ind.bollinger(df)
    squeeze = b["bandwidth"] <= b["bandwidth"].rolling(120, min_periods=120).quantile(0.2)
    return squeeze.shift(1).fillna(False).astype(bool) & ta.crossover(df["close"], b["upper"])


def _donchian_breakout(n):
    # 오늘 종가가 '어제까지' n봉 최고가를 넘음 — 터틀/돈치안/세이코타식 돌파
    return lambda df: ta.crossover(df["close"], ta.highest(df["high"], n).shift(1))


def _ma_cross(fast, slow, kind="sma"):
    f = ta.MA[kind]
    return lambda df: ta.crossover(f(df["close"], fast), f(df["close"], slow))


def _macd_cross(below_zero=False):
    def s(df):
        m = ind.macd(df)
        x = ta.crossover(m["macd"], m["signal"])
        return x & (m["macd"] < 0) if below_zero else x
    return s


def _stoch_cross_up(df):
    st = ind.stoch(df)
    return ta.crossover(st["k"], st["d"]) & (st["k"] < 20)


def _stochrsi_cross_up(df):
    st = ind.stoch_rsi(df)
    return ta.crossover(st["k"], st["d"]) & (st["k"] < 20)


def _supertrend_flip_up(df):
    d = ind.supertrend(df)["direction"]
    return (d == -1) & (d.shift(1) == 1)


def _adx_di_cross(df):
    x = ind.dmi(df)
    return ta.crossover(x["plus_di"], x["minus_di"]) & (x["adx"] > 20)


def _ichimoku_tk_above_cloud(df):
    i = ind.ichimoku(df)
    top = pd.concat([i["cloud_a"], i["cloud_b"]], axis=1).max(axis=1, skipna=False)
    return ta.crossover(i["tenkan"], i["kijun"]) & (df["close"] > top)


def _psar_flip_up(df):
    p = ind.psar(df)
    above = df["close"] > p
    return above & ~above.shift(1, fill_value=True)


def _cci_cross_up(df):
    return ta.crossover(ind.cci(df), -100)


def _mfi_cross_up(df):
    return ta.crossover(ind.mfi(df), 20)


def _williams_cross_up(df):
    return ta.crossover(ind.williams_r(df), -80)


def _connors_rsi2(df):
    # Larry Connors: 장기 상승추세(200SMA 위) 속 단기 과매도(RSI(2) < 10)
    return (df["close"] > ta.sma(df["close"], 200)) & (ind.rsi(df, 2) < 10)


def _ribbon_aligned(df):
    r = ind.ma_ribbon(df, (5, 10, 20, 60, 120))
    aligned = (r.diff(axis=1, periods=-1).iloc[:, :-1] > 0).all(axis=1) & r.notna().all(axis=1)
    return aligned & ~aligned.shift(1, fill_value=True)


def _pivot_low_confirmed(left=5, right=5):
    # '저점신호' 류: 피벗은 right 봉 뒤에야 확정 → 확정 봉에서만 신호
    return lambda df: ta.pivotlow(df["low"], left, right).notna()


BUY: dict[str, tuple[str, str, Signal]] = {
    "rsi_cross_up_30":       ("모멘텀", "Wilder RSI", _rsi_cross_up(30)),
    "stoch_cross_up_<20":    ("모멘텀", "Lane Stochastic", _stoch_cross_up),
    "stochrsi_cross_up_<20": ("모멘텀", "Chande/Kroll StochRSI", _stochrsi_cross_up),
    "macd_cross":            ("모멘텀", "Appel MACD", _macd_cross()),
    "macd_cross_below_0":    ("모멘텀", "Appel MACD", _macd_cross(True)),
    "cci_cross_up_-100":     ("모멘텀", "Lambert CCI", _cci_cross_up),
    "williams_cross_up_-80": ("모멘텀", "Williams %R", _williams_cross_up),
    "mfi_cross_up_20":       ("거래량", "MFI", _mfi_cross_up),
    "bb_lower_reentry":      ("변동성", "Bollinger 평균회귀", _bb_lower_reentry),
    "bb_squeeze_breakout":   ("변동성", "Bollinger 스퀴즈", _bb_squeeze_breakout),
    "donchian_breakout_20":  ("추세", "Donchian/Turtle S1", _donchian_breakout(20)),
    "donchian_breakout_55":  ("추세", "Turtle S2", _donchian_breakout(55)),
    "ema_cross_15_150":      ("추세", "Seykota식 EMA 교차", _ma_cross(15, 150, "ema")),
    "golden_cross_50_200":   ("추세", "골든크로스", _ma_cross(50, 200)),
    "golden_cross_20_60":    ("추세", "국내 관례 20/60", _ma_cross(20, 60)),
    "ma_ribbon_aligned":     ("추세", "정배열 전환 5>10>20>60>120", _ribbon_aligned),
    "supertrend_flip_up":    ("추세", "Supertrend", _supertrend_flip_up),
    "dmi_cross_adx>20":      ("추세", "Wilder DMI/ADX", _adx_di_cross),
    "psar_flip_up":          ("추세", "Wilder Parabolic SAR", _psar_flip_up),
    "ichimoku_tk_above_kumo": ("추세", "일목균형표", _ichimoku_tk_above_cloud),
    "connors_rsi2":          ("평균회귀", "Connors RSI(2)", _connors_rsi2),
    "pivot_low_5_5":         ("가격구조", "피벗 저점(확정 시점)", _pivot_low_confirmed()),
}


SELL: dict[str, tuple[str, str, Signal]] = {
    "rsi_cross_down_70":     ("모멘텀", "Wilder RSI", lambda df: ta.crossunder(ind.rsi(df), 70)),
    "macd_cross_down":       ("모멘텀", "Appel MACD",
                              lambda df: (lambda m: ta.crossunder(m["macd"], m["signal"]))(ind.macd(df))),
    "bb_upper_reject":       ("변동성", "Bollinger",
                              lambda df: ta.crossunder(df["close"], ind.bollinger(df)["upper"])),
    "donchian_exit_10":      ("추세", "Turtle S1 청산",
                              lambda df: ta.crossunder(df["close"], ta.lowest(df["low"], 10).shift(1))),
    "donchian_exit_20":      ("추세", "Turtle S2 청산",
                              lambda df: ta.crossunder(df["close"], ta.lowest(df["low"], 20).shift(1))),
    "ema_cross_down_15_150": ("추세", "Seykota식 EMA 교차",
                              lambda df: ta.crossunder(ta.ema(df["close"], 15), ta.ema(df["close"], 150))),
    "supertrend_flip_down":  ("추세", "Supertrend",
                              lambda df: (lambda d: (d == 1) & (d.shift(1) == -1))(ind.supertrend(df)["direction"])),
    "close_below_sma20":     ("추세", "20일선 이탈",
                              lambda df: ta.crossunder(df["close"], ta.sma(df["close"], 20))),
}


def compute(df: pd.DataFrame, catalog: dict = BUY) -> pd.DataFrame:
    return pd.DataFrame({k: v[2](df).fillna(False).astype(bool) for k, v in catalog.items()}, index=df.index)
