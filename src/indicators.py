from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
import pandas as pd

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

    # Volume (zaten varsa normalize et, yoksa 0 ata)
    if "Volume" in df.columns:
        df["Volume_Norm"] = (df["Volume"] - df["Volume"].mean()) / df["Volume"].std()
    else:
        df["Volume_Norm"] = 0

    return df
