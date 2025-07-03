# src/multi_backtest.py
import pandas as pd
from src.model_utils import load_model
from src.indicators import add_indicators, add_advanced_features
from src.backtest import backtest
import yfinance as yf

FEATURES = [
    "RSI", "EMA20", "SMA50", "MACD", "MACD_Signal", "BB_Middle",
    "BB_Upper", "BB_Lower", "Volume_Norm", "Price_Change_Pct",
    "RSI_Overbought", "RSI_Oversold", "Trend_Crossover",
    "Volatility", "MACD_Buy_Signal"
]

def run_backtest_for_ticker(ticker, period="6mo"):
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = add_indicators(df)
    df = add_advanced_features(df)
    df = df.dropna()

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    model = load_model()
    df["Prediction"] = model.predict(df[FEATURES])
    result = df[["Close", "Prediction"]]
    
    transactions = backtest(result)
    return transactions

def multi_backtest(tickers):
    summary = []

    for ticker in tickers:
        print(f"\n🔄 {ticker} backtest başlıyor...")
        try:
            transactions = run_backtest_for_ticker(ticker)
            if not transactions.empty:
                first = transactions.iloc[0]["Price"]
                last = transactions.iloc[-1]["Price"]
                gain = ((last - first) / first) * 100
                summary.append({
                    "Ticker": ticker,
                    "First": round(first, 2),
                    "Last": round(last, 2),
                    "Gain %": round(gain, 2),
                    "Trades": len(transactions)
                })
        except Exception as e:
            print(f"⚠️ {ticker} için hata: {e}")
    
    return pd.DataFrame(summary)
