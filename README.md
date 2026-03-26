# BenderQuant

A Python project that performs technical analysis on AI and semiconductor stocks using multiple machine learning models, with transparent decision explanations via SHAP.

## Overview

BenderQuant trains a classification model to predict whether a stock will gain **+5% or more within 30 days** (Buy = 1, Don't Buy = 0). It compares five ML algorithms, applies SMOTE for class balancing, and backtests the best model with stop-loss/take-profit logic.

## Tracked Stocks

| Ticker | Company |
|--------|---------|
| NVDA | NVIDIA Corporation |
| AMD | Advanced Micro Devices |
| MRVL | Marvell Technology |
| ACLS | Axcelis Technologies |
| ON | ON Semiconductor |

## Features

- Automated data fetching via `yfinance` (5-year daily OHLCV)
- Technical indicators: RSI, EMA20, SMA50, MACD, Bollinger Bands, OBV, ROC, Volume
- Binary classification: Buy / Don't Buy (30-day horizon, +5% threshold)
- Five ML models: XGBoost, Random Forest, LightGBM, CatBoost, Logistic Regression
- SMOTE oversampling for class imbalance
- SHAP explainability (beeswarm + waterfall plots)
- Hyperparameter tuning via GridSearchCV (108 candidates, 5-fold)
- Stratified 5-fold cross-validation
- Backtest engine with stop-loss (5%) and take-profit (10%)
- Equity curve and trade log export per ticker

## Model Results

### Cross-Validation — f1_weighted (5-fold, stratified)

| Model | Before SMOTE | After SMOTE |
|-------|-------------|------------|
| **Random Forest** | **0.9017 ± 0.0128** | **0.9147 ± 0.0068** |
| XGBoost | 0.8802 ± 0.0064 | 0.8977 ± 0.0088 |
| LightGBM | 0.8664 ± 0.0104 | 0.8842 ± 0.0033 |
| CatBoost | 0.8627 ± 0.0094 | 0.8812 ± 0.0052 |
| Logistic Regression | 0.5986 ± 0.0113 | 0.6071 ± 0.0142 |

### XGBoost Test Set (hold-out 30%)

| Metric | Don't Buy | Buy |
|--------|-----------|-----|
| Precision | 0.87 | 0.89 |
| Recall | 0.93 | 0.82 |
| F1 | 0.90 | 0.85 |
| **Accuracy** | **0.88** | — |

### Best Hyperparameters (GridSearchCV)

```
colsample_bytree: 1.0
learning_rate:    0.1
max_depth:        7
n_estimators:     150
subsample:        1.0
```

### Top SHAP Features (mean absolute SHAP value)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | SMA50 | 0.919 |
| 2 | BB_Upper | 0.481 |
| 3 | Symbol | 0.462 |
| 4 | MACD_Signal | 0.434 |
| 5 | BB_Lower | 0.423 |
| 6 | EMA20 | 0.292 |
| 7 | MACD | 0.272 |
| 8 | Volatility | 0.271 |
| 9 | RSI | 0.253 |
| 10 | BB_Middle | 0.236 |

## Backtest Results — NVDA Sample (Apr–Aug 2025)

Starting capital: $10,000

| Trade | Action | Price |
|-------|--------|-------|
| 1 | BUY | $112.19 |
| 1 | SELL | $104.48 |
| 2 | BUY | $101.48 |
| 2 | SELL | $114.49 |
| 3 | BUY | $113.81 |
| 3 | SELL | $129.92 |
| 4 | BUY | $135.33 |
| 4 | SELL | $154.31 |

**Final equity: $14,310.99 (+43.1% over ~4 months)**

> Note: Backtest does not account for transaction costs, slippage, or spread. Results reflect simulated performance on historical data.

## Known Limitations

- **Look-ahead bias**: Target labels use 30-day future returns. Backtest predictions are made on data the model was trained on — results are not walkforward validated.
- **Small dataset**: ~6,000 rows across 5 tickers (5 years daily). Tree models may overfit.
- **Sector bias**: All 5 tickers are AI/semiconductor stocks that experienced strong bull runs in the training period (2020–2024). The model may have learned sector momentum rather than generalizable signals.
- **No transaction costs**: Commission, spread, and slippage are not modeled.

## Project Structure

```
benderquant/
├── src/
│   ├── data_loader.py          # yfinance data fetching
│   ├── indicators.py           # technical indicator calculation
│   ├── labeling.py             # target label generation
│   ├── dataset_builder.py      # multi-ticker dataset assembly
│   ├── model_train.py          # XGBoost training + SHAP
│   ├── model_tuning.py         # GridSearchCV
│   ├── model_compare.py        # 5-model comparison
│   ├── cross_validate_model.py # stratified CV
│   ├── oversample.py           # SMOTE
│   ├── backtest.py             # single-ticker backtest engine
│   ├── multi_backtest.py       # multi-ticker backtest
│   ├── metrics.py              # evaluation metrics
│   ├── plotter.py              # indicator charts
│   ├── plot_equity.py          # equity curve plots
│   ├── summary.py              # yfinance stock info
│   └── config.py               # ticker list
├── data/                       # CSV data + summary files
├── models/                     # saved model + label encoder
├── logs/                       # trade logs + equity curves
├── main.py                     # full training pipeline
├── inference.py                # prediction on new data
├── streamlit_app.py            # interactive dashboard
└── multi_backtest_run.py       # run backtest for all tickers
```

## Installation

```bash
git clone https://github.com/yusufbender/benderquant.git
cd benderquant
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Dependencies

```
pandas, numpy, yfinance, scikit-learn
xgboost, lightgbm, catboost
matplotlib, seaborn, shap
imbalanced-learn, streamlit, joblib
```

## License

MIT