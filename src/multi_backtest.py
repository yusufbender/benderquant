import pandas as pd
import yfinance as yf
from src.model_utils import load_model
from src.indicators import add_indicators, add_advanced_features
from src.backtest import backtest
from src.metrics import summarize_performance
from src.plot_equity import plot_equity_curve

import os
os.makedirs("logs", exist_ok=True)

# Modelin beklediği tüm feature listesi (Symbol dahil!)
FEATURES = [
    "RSI", "EMA20", "SMA50", "MACD", "MACD_Signal", "BB_Middle",
    "BB_Upper", "BB_Lower", "Volume_Norm", "Price_Change_Pct",
    "RSI_Overbought", "RSI_Oversold", "Trend_Crossover",
    "Volatility", "MACD_Buy_Signal", "Symbol"
]

def run_backtest_for_ticker(ticker, period="6mo"):
    df = yf.download(ticker, period=period, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = add_indicators(df)
    df = add_advanced_features(df)

    df["Symbol"] = ticker

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    df = df.dropna()

    model = load_model()
    from joblib import load
    le = load("models/symbol_encoder.pkl")
    df["Symbol"] = le.transform(df["Symbol"])

    df["Prediction"] = model.predict(df[FEATURES])
    result = df[["Close", "Prediction"]]

    # ✅ Güncellenmiş: Yeni backtest çift çıktı döner
    trades_df, equity_df = backtest(result, stop_loss=0.05, take_profit=0.1)
    return trades_df, equity_df


def multi_backtest(tickers):
    summary = []

    for ticker in tickers:
        print(f"\n🔄 {ticker} backtest başlıyor...")
        try:
            trades_df, equity_df = run_backtest_for_ticker(ticker)
            if not trades_df.empty:
                first = trades_df.iloc[0]["Price"]
                last = trades_df.iloc[-1]["Price"]
                gain = ((last - first) / first) * 100

                # 📈 Risk metrikleri özetini al
                performance = summarize_performance(equity_df)

                # 📊 Özeti genişlet
                summary.append({
                    "Ticker": ticker,
                    "First": round(first, 2),
                    "Last": round(last, 2),
                    "Gain %": round(gain, 2),
                    "Trades": len(trades_df),
                    "Sharpe": performance["Sharpe Ratio"],
                    "Max DD %": performance["Max Drawdown (%)"],
                    "CAGR %": performance["CAGR (%)"]
                })

                # CSV export
                equity_df.to_csv(f"logs/{ticker}_equity.csv")
                trades_df.to_csv(f"logs/{ticker}_trades.csv")
                plot_equity_curve(equity_df, trades_df, save_path=f"logs/{ticker}_equity.png")

        except Exception as e:
            print(f"⚠️ {ticker} için hata: {e}")

    return pd.DataFrame(summary)
