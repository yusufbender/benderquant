# streamlit_app.py

import streamlit as st
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

st.set_page_config(page_title="📈 BenderQuant", layout="wide")
st.title("🤖 BenderQuant Tahmin ve Backtest Platformu")

ticker = st.text_input("🎯 Hisse Senedi Sembolü (örnek: NVDA)", "NVDA")
period = st.selectbox("⏳ Veri Aralığı", ["3mo", "6mo", "1y", "2y"], index=1)

if st.button("🚀 Tahmin ve Backtest Başlat"):
    model = load_model()
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = add_indicators(df)
    df = add_advanced_features(df)
    df = df.dropna()

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    preds = model.predict(df[FEATURES])
    df["Prediction"] = preds
    df["Signal"] = df["Prediction"].map({1: "📈 AL", 0: "📉 SAT"})

    st.subheader("📉 Hisse Tahminleri")
    st.dataframe(df[["Close", "Prediction", "Signal"]].tail(30), use_container_width=True)

    st.subheader("💰 Backtest Sonuçları")
    transactions = backtest(df[["Close", "Prediction"]])
    st.dataframe(transactions, use_container_width=True)
