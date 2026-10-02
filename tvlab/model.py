"""지표들을 묶어 'h봉 뒤 상승 확률'을 추정하는 워크포워드 모델.

단일 신호의 승률표(evaluate.event_study)가 1단계라면, 이건 2단계다.
지표 값을 **스케일 무관한 특징**(가격 대비 거리, 0~100 오실레이터 등)으로 바꾸고
로지스틱 회귀로 P(close[t+h] > open[t+1]) 를 추정한다.

* 확장 윈도 워크포워드 + **purge gap = h** (라벨이 겹치는 구간을 학습에서 제거)
* 출력: 폴드별 AUC, 그리고 예측확률 구간별 실제 적중률(보정 곡선)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind
from . import ta
from .evaluate import forward_returns


def features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    a = ind.atr(df, 14)
    bb = ind.bollinger(df)
    m = ind.macd(df)
    d = ind.dmi(df)
    st = ind.stoch(df)
    dc = ind.donchian(df, 20)
    X = pd.DataFrame(index=df.index)
    X["rsi14"] = ind.rsi(df) / 100
    X["rsi2"] = ind.rsi(df, 2) / 100
    X["stoch_k"] = st["k"] / 100
    X["bb_pct_b"] = bb["pct_b"]
    X["bb_width_rank"] = bb["bandwidth"].rolling(250, min_periods=60).rank(pct=True)
    X["macd_hist_atr"] = m["hist"] / a
    X["adx"] = d["adx"] / 100
    X["di_diff"] = (d["plus_di"] - d["minus_di"]) / 100
    X["dc_pos"] = (c - dc["lower"]) / (dc["upper"] - dc["lower"])
    for n in (20, 60, 200):
        X[f"dist_sma{n}_atr"] = (c - ta.sma(c, n)) / a
    X["ret5_atr"] = c.diff(5) / a
    X["supertrend_up"] = (ind.supertrend(df)["direction"] == -1).astype(float)
    if df["volume"].notna().any():
        X["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
        X["mfi"] = ind.mfi(df) / 100
    return X.replace([np.inf, -np.inf], np.nan)


def walk_forward(df: pd.DataFrame, h: int = 10, n_folds: int = 5, min_train: int = 750) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = features(df)
    y = (forward_returns(df, h) > 0).astype(float).where(forward_returns(df, h).notna())
    data = X.join(y.rename("y")).dropna()
    Xv, yv, idx = data.drop(columns="y").to_numpy(), data["y"].to_numpy(), data.index
    n = len(data)
    if n < min_train + n_folds * 50:
        raise ValueError(f"데이터 부족: 유효 표본 {n}개")
    edges = np.linspace(min_train, n, n_folds + 1).astype(int)
    preds = pd.Series(np.nan, index=idx)
    folds = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        tr_end = lo - h  # purge
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        clf.fit(Xv[:tr_end], yv[:tr_end])
        p = clf.predict_proba(Xv[lo:hi])[:, 1]
        preds.iloc[lo:hi] = p
        auc = roc_auc_score(yv[lo:hi], p) if len(set(yv[lo:hi])) > 1 else np.nan
        folds.append({"test_start": idx[lo], "test_end": idx[hi - 1], "auc": auc,
                      "base_rate": float(yv[lo:hi].mean())})
    coef = pd.Series(clf[-1].coef_[0], index=X.columns).sort_values()
    oos = pd.DataFrame({"p": preds, "y": data["y"]}).dropna()
    oos["bucket"] = pd.qcut(oos["p"], 5, labels=["Q1(낮음)", "Q2", "Q3", "Q4", "Q5(높음)"], duplicates="drop")
    calib = oos.groupby("bucket", observed=True).agg(n=("y", "size"), mean_p=("p", "mean"), hit=("y", "mean"))
    return {"folds": pd.DataFrame(folds), "calibration": calib, "coef_last_fold": coef, "pred": preds}
