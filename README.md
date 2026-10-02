# trading-playground / tvlab

TradingView 지표를 **소수점까지 똑같이 재현**하고, 각 지표의 매수/매도 신호가
**기저율보다 실제로 나은지** 통계적으로 검증하는 파이썬 툴킷.

지표별 해설과 검증 방법론: [docs/indicators.md](docs/indicators.md)

## 설치

```bash
pip install -r requirements.txt          # pandas numpy scipy scikit-learn pytest
pip install finance-datareader           # (선택) 국내 종목 무료 시세
pip install yfinance                     # (선택) 해외 티커
```

## 워크플로

### 1단계 — 데이터 받기
* **무료**: `python -m tvlab --krx 005930` (FinanceDataReader). 지표는 tvlab 이 Pine 과
  같은 규칙으로 직접 계산하므로 TradingView 내보내기가 없어도 분석은 그대로 된다.
* **TradingView 유료 플랜**: 차트 **⋯ → Export chart data** 로 가격과 켜진 모든 플롯을
  CSV 로 받을 수 있다. 커스텀 지표의 수식을 소수점까지 대조할 때만 필요하다.

### 2단계 — 지표 수식 확인 (커스텀 지표 역추적)
```python
from tvlab import data, fit
df = data.load_tradingview_csv("KRX_005930_1D.csv")
fit.fit_line(df, df["MA 1"])                                # → sma(close,5) ?
fit.fit_band(df, upper=df["Upper"], lower=df["Lower"])      # 중심/반폭 분해
fit.is_donchian(df, df["Upper"])                            # 돈치안이면 오차≈0
```

### 3단계 — 신호 승률표
```bash
python -m tvlab --csv KRX_005930_1D.csv --horizons 5,10,20 --out result.csv
python -m tvlab --krx 005930 --model     # 무료 국내 시세 + 워크포워드 확률 모델
python -m tvlab --demo                   # 합성 데이터로 파이프라인 확인
```

출력 컬럼:

| 컬럼 | 의미 |
|---|---|
| `n` | 겹침 제거 후 신호 수 |
| `win_rate` / `base_win` | 신호 후 h봉 수익>0 비율 / 아무 날이나 샀을 때 |
| `edge` | 신호 평균수익 − 기저 평균수익 |
| `is_edge` / `oos_edge` | 전반 60% / 후반 40% 각각의 edge |
| `q_value` | 다중검정(BH) 보정 p값 |
| `consistent` | 전·후반 모두 edge>0, n≥20 |
| `robust` | consistent 이면서 q<0.10 — **여기 True 인 것만 믿을 것** |

### 4단계 — 매수+매도 조합 백테스트
```python
from tvlab import evaluate, signals
buy  = signals.BUY["donchian_breakout_20"][2](df)
sell = signals.SELL["donchian_exit_10"][2](df).fillna(False)
trades, metrics = evaluate.backtest(df, buy, sell, atr_stop=2.0, cost=0.0025)
```

## 구성

```
tvlab/ta.py          Pine ta.* 재현 (sma/ema/rma/stdev/atr/pivot/crossover …)
tvlab/indicators.py  인기 지표 20종 (TradingView 기본 파라미터)
tvlab/signals.py     매수 22종 / 매도 8종 규칙 카탈로그
tvlab/evaluate.py    이벤트 스터디, IS/OOS, BH 보정, 백테스트
tvlab/fit.py         TV 플롯 → 수식 역추적 (중심/반폭 분해)
tvlab/model.py       지표 결합 워크포워드 확률 모델
tests/               Pine 규칙 · 미래참조 없음 · 역추적 재현 테스트
```

## 원칙
* 모든 신호는 t 종가 확정, t+1 시가 진입. 테스트가 전 신호의 미래참조 여부를 검사한다.
* 랜덤워크에서 `robust` 신호가 0개여야 한다 (테스트 포함). 실제 데이터에서도 대부분은 0개다 — 그게 정상.
* 이 저장소는 연구용이며 투자 조언이 아니다.
