from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator

def add_indicators(df):
    if "Close" not in df.columns or df["Close"].isnull().all():
        raise ValueError("Veri 'Close' sütunu içermiyor ya da tümü boş.")
    
    close = df["Close"].astype(float).squeeze()

    df["RSI"] = RSIIndicator(close=close).rsi()
    df["EMA20"] = EMAIndicator(close=close, window=20).ema_indicator()
    df["SMA50"] = SMAIndicator(close=close, window=50).sma_indicator()
    return df
