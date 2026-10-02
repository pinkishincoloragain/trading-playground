"""사용법:

    python -m tvlab --csv BINANCE_BTCUSDT_1D.csv
    python -m tvlab --ticker 005930.KS          # yfinance 설치 시
    python -m tvlab --demo                      # 합성 데이터 (파이프라인 확인용)
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import data, evaluate, signals


def main() -> None:
    ap = argparse.ArgumentParser(prog="tvlab")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", help="TradingView 'Export chart data' CSV")
    g.add_argument("--ticker", help="yfinance 티커 (예: SPY, 005930.KS)")
    g.add_argument("--demo", action="store_true", help="합성 랜덤워크 데이터")
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--split", type=float, default=0.6, help="IS/OOS 분할 비율")
    ap.add_argument("--model", action="store_true", help="워크포워드 확률 모델도 실행")
    ap.add_argument("--out", help="결과 CSV 저장 경로")
    a = ap.parse_args()

    df = (data.load_tradingview_csv(a.csv) if a.csv
          else data.load_yfinance(a.ticker) if a.ticker else data.synthetic())
    hs = tuple(int(x) for x in a.horizons.split(","))
    print(f"bars={len(df)}  {df.index[0]} → {df.index[-1]}")

    res = evaluate.event_study(df, signals.compute(df, signals.BUY), hs, split=a.split)
    cols = ["signal", "h", "n", "win_rate", "base_win", "mean", "base_mean", "edge",
            "is_edge", "oos_edge", "q_value", "consistent", "robust"]
    pd.set_option("display.width", 200, "display.max_rows", 200)
    fmt = {c: "{:.2%}".format for c in ["win_rate", "base_win", "mean", "base_mean", "edge", "is_edge", "oos_edge"]}
    print(res[cols].to_string(formatters=fmt, float_format="{:.3f}".format))
    if a.out:
        res.to_csv(a.out, index=False)

    if a.model:
        from . import model
        r = model.walk_forward(df, h=hs[len(hs) // 2])
        print("\n[walk-forward folds]\n", r["folds"].to_string())
        print("\n[calibration: 예측확률 5분위별 실제 적중률]\n", r["calibration"].to_string())


if __name__ == "__main__":
    main()
