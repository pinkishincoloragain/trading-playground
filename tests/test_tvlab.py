import numpy as np
import pandas as pd
import pytest

from tvlab import data, evaluate, fit, indicators as ind, signals, ta


@pytest.fixture(scope="module")
def df():
    return data.synthetic(1500, seed=1)


def test_ema_seeded_with_sma(df):
    c = df["close"]
    e = ta.ema(c, 10)
    assert e.iloc[:9].isna().all()
    assert e.iloc[9] == pytest.approx(c.iloc[:10].mean())
    a = 2 / 11
    assert e.iloc[10] == pytest.approx(a * c.iloc[10] + (1 - a) * e.iloc[9])


def test_rma_alpha(df):
    c = df["close"]
    r = ta.rma(c, 14)
    assert r.iloc[14] == pytest.approx(c.iloc[14] / 14 + r.iloc[13] * 13 / 14)


def test_stdev_is_population(df):
    c = df["close"]
    assert ta.stdev(c, 20).iloc[19] == pytest.approx(np.std(c.iloc[:20].to_numpy(), ddof=0))


def test_rsi_bounds(df):
    r = ind.rsi(df).dropna()
    assert r.between(0, 100).all()


def test_supertrend_direction_values(df):
    d = ind.supertrend(df)["direction"].dropna()
    assert set(d.unique()) <= {-1.0, 1.0}


def test_pivotlow_marked_on_confirmation_bar():
    low = pd.Series([5, 4, 3, 2, 1, 2, 3, 4, 5], dtype=float)
    p = ta.pivotlow(low, 2, 2)
    assert p.notna().sum() == 1
    assert p.iloc[6] == 1  # 피벗 봉(4) + right(2)


@pytest.mark.parametrize("name", list(signals.BUY) + list(signals.SELL))
def test_signals_have_no_lookahead(df, name):
    """t 까지의 데이터만으로 계산한 신호 == 전체 데이터로 계산한 신호의 t 값."""
    fn = {**signals.BUY, **signals.SELL}[name][2]
    full = fn(df).fillna(False).astype(bool)
    for t in (700, 1000, 1300):
        part = fn(df.iloc[: t + 1]).fillna(False).astype(bool)
        assert bool(part.iloc[-1]) == bool(full.iloc[t]), f"{name} @ {t}"


def test_fit_band_recovers_bollinger_30(df):
    """사용자 사례 재현: 중심/반폭 분해로 sma(close,30) ± 2·stdev(close,30) 를 찾는다."""
    bb = ind.bollinger(df, length=30, mult=2)
    res = fit.fit_band(df, bb["upper"], bb["lower"], lengths=range(10, 41), sources=("close", "hl2"))
    best = res.iloc[0]
    assert best["center"] == "sma(close,30)"
    assert "stdev(close,30)" in best["width"] and "sample" not in best["width"]
    assert best["k"] == pytest.approx(2.0)
    assert best["width_mape"] < 1e-9
    n, err = fit.is_donchian(df, bb["upper"], lengths=range(10, 41))
    assert err > 1e-3  # 돈치안 탈락


def test_random_walk_has_no_robust_signal():
    df = data.synthetic(3000, seed=0)
    res = evaluate.event_study(df, signals.compute(df), (10,))
    assert not res["robust"].any()


def test_backtest_runs(df):
    sig = signals.compute(df)
    tr, m = evaluate.backtest(df, sig["donchian_breakout_20"],
                              signals.SELL["donchian_exit_10"][2](df).fillna(False), atr_stop=2)
    assert m["trades"] > 0 and -1 < m["max_dd"] <= 0
