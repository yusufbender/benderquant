BenderQuant
A Python project that performs technical analysis on AI and semiconductor stocks using multiple machine learning models, with transparent decision explanations via SHAP.
Overview
BenderQuant trains a binary classifier to predict whether a stock will gain +5% or more within 30 days (Buy = 1, Don't Buy = 0). It compares five ML algorithms, applies SMOTE for class balancing, runs hyperparameter tuning, and evaluates with both stratified cross-validation and walkforward validation.
Tracked Stocks
Ticker	Company
NVDA	NVIDIA Corporation
AMD	Advanced Micro Devices
MRVL	Marvell Technology
ACLS	Axcelis Technologies
ON	ON Semiconductor
Features
Automated data fetching via `yfinance` (5-year daily OHLCV)
Technical indicators: RSI, EMA20, SMA50, MACD, Bollinger Bands, OBV, ROC, Volume
Binary classification: Buy / Don't Buy (30-day horizon, +5% threshold)
Five ML models: XGBoost, Random Forest, LightGBM, CatBoost, Logistic Regression
SMOTE oversampling for class imbalance
SHAP explainability (beeswarm + waterfall plots)
Hyperparameter tuning via GridSearchCV (108 candidates, 5-fold)
Stratified 5-fold cross-validation
Walkforward (expanding window) validation — out-of-sample, time-aware evaluation
Backtest engine with stop-loss (5%) and take-profit (10%)
Equity curve and trade log export per ticker
Model Results (v2 — Symbol feature removed)
Cross-Validation — f1_weighted (5-fold, stratified)
Model	Before SMOTE	After SMOTE
Random Forest	0.8701 ± 0.0110	0.8956 ± 0.0067
XGBoost	0.8428 ± 0.0116	0.8692 ± 0.0048
LightGBM	0.8326 ± 0.0130	0.8485 ± 0.0102
CatBoost	0.8270 ± 0.0145	0.8431 ± 0.0011
Logistic Regression	0.5996 ± 0.0151	0.5782 ± 0.0117
XGBoost Test Set (hold-out 30%)
Metric	Don't Buy	Buy
Precision	0.85	0.87
Recall	0.91	0.78
F1	0.88	0.82
Accuracy	0.86	—
Best Hyperparameters (GridSearchCV)
```
colsample_bytree: 1.0
learning_rate:    0.2
max_depth:        7
n_estimators:     150
subsample:        1.0
```
Top SHAP Features (v2 — Symbol removed)
Rank	Feature	Importance
1	SMA50	0.663
2	BB_Upper	0.662
3	BB_Lower	0.415
4	BB_Middle	0.352
5	Volatility	0.325
6	MACD_Signal	0.309
7	Volume_Norm	0.293
8	MACD	0.273
9	EMA20	0.266
10	RSI	0.200
Walkforward Validation
Walkforward (expanding window) validation is a time-aware evaluation method. Unlike stratified cross-validation, the model is always trained on past data and tested on future data it has never seen — which is closer to real trading conditions.
```
Total data: ~5875 rows (time-ordered)

Train 1: [0 → 3525]   Test 1: [3525 → 3995]
Train 2: [0 → 3995]   Test 2: [3995 → 4465]  ← expanding window
Train 3: [0 → 4465]   Test 3: [4465 → 4935]
Train 4: [0 → 4935]   Test 4: [4935 → 5405]
Train 5: [0 → 5405]   Test 5: [5405 → 5875]
```
Walkforward Results (v2)
Fold	Train rows	Test rows	F1 (weighted)	Precision	Recall	Buy signal %
1	3525	470	0.3216	0.2395	0.4894	100.0%
2	3995	470	0.5556	0.5669	0.5638	36.4%
3	4465	470	0.6073	0.6292	0.6234	29.6%
4	4935	470	0.4938	0.5729	0.5000	66.2%
5	5405	470	0.5126	0.5299	0.5021	42.6%
Avg	—	—	0.4982 ± 0.1079	0.5077	0.5357	55.0%
Stratified CV vs Walkforward
Method	F1 (weighted)
Stratified 5-fold CV	0.8428
Walkforward (time-aware)	0.4982
The ~0.34 gap between stratified CV and walkforward F1 is the clearest signal of look-ahead bias in the training pipeline. Stratified CV shuffles data randomly, allowing the model to indirectly learn from future patterns. Walkforward enforces strict temporal ordering — the model only ever sees the past when making predictions about the future.
Fold 1's 100% Buy signal rate suggests the model trained on mixed-ticker data does not generalize well to the earliest time window, likely due to distributional differences between the 2020–2024 bull market training period and the out-of-sample test window.
Backtest Results — NVDA Sample (Apr–Aug 2025)
Starting capital: $10,000
Trade	Action	Price
1	BUY	$112.19
1	SELL	$104.48
2	BUY	$101.48
2	SELL	$114.49
3	BUY	$113.81
3	SELL	$129.92
4	BUY	$135.33
4	SELL	$154.31
Final equity: $14,310.99 (+43.1% over ~4 months)
> Note: Backtest does not account for transaction costs, slippage, or spread. Results reflect simulated performance on historical data and should not be used for real trading decisions.
Known Limitations
Look-ahead bias: Target labels use 30-day future returns (`shift(-30)`). The walkforward results (F1 ≈ 0.50) reflect a more realistic performance estimate than stratified CV (F1 ≈ 0.84).
Small dataset: ~5,875 rows across 5 tickers (5 years daily). Tree models may overfit.
Sector bias: All 5 tickers are AI/semiconductor stocks that experienced a strong bull run during 2020–2024. The model may have learned sector momentum rather than generalizable technical signals.
No transaction costs: Commission, spread, and slippage are not modeled.
No per-ticker walkforward: Current walkforward mixes all 5 tickers in a single time-ordered dataset, which does not perfectly reflect per-asset trading logic.
Changelog
v2 (current)
Removed `Symbol` as a model feature (was 3rd in SHAP importance — potential leakage)
Added walkforward (expanding window) validation
Updated README with honest performance comparison
v1
Initial release with stratified CV only
Symbol encoded and used as a feature
Project Structure
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
│   ├── walkforward.py          # expanding window validation
│   ├── oversample.py           # SMOTE
│   ├── backtest.py             # single-ticker backtest engine
│   ├── multi_backtest.py       # multi-ticker backtest
│   ├── metrics.py              # evaluation metrics
│   ├── plotter.py              # indicator charts
│   ├── plot_equity.py          # equity curve plots
│   ├── summary.py              # yfinance stock info
│   └── config.py               # ticker list
├── data/                       # CSV data + summary files
├── models/                     # saved model pkl
├── logs/                       # trade logs + equity curves
├── main.py                     # full training pipeline
├── inference.py                # prediction on new data
├── streamlit_app.py            # interactive dashboard
└── multi_backtest_run.py       # run backtest for all tickers
```
Installation
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
Dependencies
```
pandas, numpy, yfinance, scikit-learn
xgboost, lightgbm, catboost
matplotlib, seaborn, shap
imbalanced-learn, streamlit, joblib
```
License
MIT