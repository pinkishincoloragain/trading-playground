"""신호가 '확률 높은 매수 타이밍'인지 검증한다.

핵심 질문은 "신호 뒤 수익률이 플러스였나?"가 아니라
**"아무 날이나 샀을 때(기저율)보다 나았나? 그리고 표본 밖에서도 그랬나?"** 이다.

* 진입: 신호 봉 다음 봉 시가 (``open[t+1]``)
* 성과: ``close[t+h] / open[t+1] - 1``
* 기저율: 같은 기간 모든 봉에서 같은 방식으로 산 경우
* 과대평가 방지: 겹치는 신호 제거(쿨다운), 전/후반 분할(IS/OOS), BH 다중검정 보정
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def forward_returns(df: pd.DataFrame, h: int) -> pd.Series:
    entry = df["open"].shift(-1)
    return df["close"].shift(-h) / entry - 1


def declutter(sig: pd.Series, gap: int) -> pd.Series:
    """직전 채택 신호로부터 ``gap`` 봉 이내 신호는 버린다 (겹치는 표본 → 과신 방지)."""
    out = np.zeros(len(sig), dtype=bool)
    last = -10**9
    for i, v in enumerate(sig.to_numpy()):
        if v and i - last >= gap:
            out[i], last = True, i
    return pd.Series(out, index=sig.index)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - w, c + w


def _stats(r: pd.Series, base: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    if n == 0:
        return {"n": 0}
    win = int((r > 0).sum())
    lo, hi = wilson(win, n)
    base = base.dropna()
    # Welch t-test: 신호 후 수익률 vs 전체 봉 수익률
    p = stats.ttest_ind(r, base, equal_var=False).pvalue if n >= 3 else np.nan
    return {
        "n": n,
        "win_rate": win / n,
        "win_ci_lo": lo,
        "win_ci_hi": hi,
        "base_win": float((base > 0).mean()),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "base_mean": float(base.mean()),
        "edge": float(r.mean() - base.mean()),
        "p_value": float(p),
    }


def bh_adjust(p: pd.Series) -> pd.Series:
    """Benjamini–Hochberg. 신호 20개를 동시에 시험하면 운으로 1개쯤은 '유의'하다."""
    q = p.copy()
    valid = p.dropna().sort_values()
    m = len(valid)
    if m == 0:
        return q
    adj = (valid * m / np.arange(1, m + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)
    q.loc[adj.index] = adj
    return q


def event_study(df: pd.DataFrame, signals: pd.DataFrame, horizons=(5, 10, 20),
                split: float = 0.6, cooldown: bool = True) -> pd.DataFrame:
    """신호 × 보유기간별 통계. ``split`` 앞부분=IS, 뒷부분=OOS."""
    cut = df.index[int(len(df) * split)]
    rows = []
    for h in horizons:
        fr = forward_returns(df, h)
        for name in signals.columns:
            s = signals[name]
            if cooldown:
                s = declutter(s, h)
            row = {"signal": name, "h": h}
            for tag, mask in (("all", slice(None)), ("is", df.index < cut), ("oos", df.index >= cut)):
                st = _stats(fr[s & mask] if tag != "all" else fr[s], fr[mask] if tag != "all" else fr)
                row.update({f"{tag}_{k}" if tag != "all" else k: v for k, v in st.items()})
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value"] = bh_adjust(out["p_value"])
    # 일관성: 전/후반 모두에서 기저율을 이김. 견고성: 거기에 다중검정 보정 후에도 유의.
    out["consistent"] = (out["is_edge"] > 0) & (out["oos_edge"] > 0) & (out["n"] >= 20)
    out["robust"] = out["consistent"] & (out["q_value"] < 0.10)
    return out.sort_values(["h", "edge"], ascending=[True, False]).reset_index(drop=True)


def backtest(df: pd.DataFrame, entry: pd.Series, exit: pd.Series | None = None,
             atr_stop: float | None = None, max_hold: int | None = None,
             cost: float = 0.0015) -> tuple[pd.DataFrame, dict]:
    """롱 온리 1포지션 백테스트. 신호는 t 종가, 체결은 t+1 시가.

    ``atr_stop``: 진입가 − k·ATR(20) 아래로 종가가 내려가면 청산 (세이코타/터틀식 리스크 관리).
    ``cost``: 왕복 거래비용(수수료+세금+슬리피지) — 국내 주식 기준 대략 0.15~0.3%.
    """
    from .indicators import atr as _atr
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    en = entry.to_numpy()
    ex = exit.to_numpy() if exit is not None else np.zeros(len(df), bool)
    a = _atr(df, 20).to_numpy()
    trades, pos, stop = [], None, None
    for t in range(len(df) - 1):
        if pos is None:
            if en[t]:
                pos = (t + 1, o[t + 1])
                stop = o[t + 1] - atr_stop * a[t] if atr_stop and not np.isnan(a[t]) else None
            continue
        held = t - pos[0]
        if ex[t] or (stop is not None and c[t] < stop) or (max_hold and held >= max_hold):
            px = o[t + 1]
            trades.append({"entry_time": df.index[pos[0]], "exit_time": df.index[t + 1],
                           "bars": t + 1 - pos[0], "ret": px / pos[1] - 1 - cost})
            pos = None
    tr = pd.DataFrame(trades)
    if tr.empty:
        return tr, {"trades": 0}
    eq = (1 + tr["ret"]).cumprod()
    wins, losses = tr.loc[tr.ret > 0, "ret"].sum(), -tr.loc[tr.ret <= 0, "ret"].sum()
    years = max((df.index[-1] - df.index[0]).days / 365.25, 1e-9) if isinstance(df.index, pd.DatetimeIndex) else np.nan
    bh = df["close"].iloc[-1] / df["open"].iloc[0] - 1
    return tr, {
        "trades": len(tr),
        "win_rate": float((tr.ret > 0).mean()),
        "avg_ret": float(tr.ret.mean()),
        "profit_factor": float(wins / losses) if losses > 0 else np.inf,
        "total_ret": float(eq.iloc[-1] - 1),
        "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years == years else np.nan,
        "max_dd": float((eq / eq.cummax() - 1).min()),
        "exposure": float(tr.bars.sum() / len(df)),
        "buy_hold": float(bh),
    }
