from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # EMA20 ve SMA50
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # EMA / SMA oranı
    df["EMA_SMA_Ratio"] = df["EMA20"] / df["SMA50"]

    # MACD ve MACD Signal
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + 2 * std
    df["BB_Lower"] = df["BB_Middle"] - 2 * std

    # Bollinger Genişliği
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # Fiyatın Bollinger Bantlarındaki Pozisyonu (0-1 arası)
    df["BB_Pos"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    # ROC (% Değişim, 5 günlük)
    df["ROC"] = df["Close"].pct_change(periods=5) * 100

    # OBV (On-Balance Volume)
    if "Volume" in df.columns:
        df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    else:
        df["OBV"] = 0

    # Normalize edilmiş Hacim
    if "Volume" in df.columns:
        df["Volume_Norm"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
    else:
        df["Volume_Norm"] = 0

    return df

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    # Fiyat değişim yüzdesi
    df["Price_Change_Pct"] = df["Close"].pct_change()

    # RSI sinyalleri
    df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
    df["RSI_Oversold"] = (df["RSI"] < 30).astype(int)

    # EMA vs SMA crossover (trend değişimi)
    df["Trend_Crossover"] = (df["EMA20"] > df["SMA50"]).astype(int)

    # Fiyat volatilitesi (fark aralığı)
    df["Volatility"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]

    # MACD pozitif crossover (al sinyali gibi)
    df["MACD_Buy_Signal"] = ((df["MACD"] > df["MACD_Signal"]) & (df["MACD"].shift(1) <= df["MACD_Signal"].shift(1))).astype(int)

    return df

