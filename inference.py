import pandas as pd
import yfinance as yf
from src.model_utils import load_model
from src.indicators import add_indicators, add_advanced_features
from src.backtest import backtest

# ⚙️ Modelin eğitildiği özellikler
FEATURES = [
    "RSI", "EMA20", "SMA50", "MACD", "MACD_Signal", "BB_Middle",
    "BB_Upper", "BB_Lower", "Volume_Norm", "Price_Change_Pct",
    "RSI_Overbought", "RSI_Oversold", "Trend_Crossover",
    "Volatility", "MACD_Buy_Signal"
]

def load_real_data(ticker="NVDA", period="6mo"):
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print("\n📊 İlk satırlar:")
    print(df.head())
    print("\n📐 Veri Tipleri:")
    print(df.dtypes)
    print("\n🧱 Şekli:", df.shape)

    df = add_indicators(df)
    df = add_advanced_features(df)
    df = df.dropna()
    return df

# 🚀 Ana Akış
if __name__ == "__main__":
    model = load_model()
    df = load_real_data("NVDA")

    # 🧪 Eksik feature kontrolü
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[FEATURES]
    preds = model.predict(X)
    df["Prediction"] = preds
    df["Signal"] = df["Prediction"].map({1: "📈 AL", 0: "📉 SAT"})

    print(df[["Close", "Prediction", "Signal"]].tail(20))
    print(f"Toplam veri adedi: {len(df)} gün")

    # 💹 Backtest
    result = df[["Close", "Prediction"]]
    transactions = backtest(result)
    print("\n📋 İşlem Geçmişi ve Backtest Sonucu:")
    print(transactions)


from src.multi_backtest import multi_backtest

tickers = ["NVDA", "AMD", "MRVL", "ACLS", "ON"]
summary_df = multi_backtest(tickers)
print("\n📊 Çoklu Hisse Backtest Özeti:")
print(summary_df)