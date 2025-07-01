import yfinance as yf
import pandas as pd

def get_stock_data(ticker: str, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)

    # Eğer sütunlar MultiIndex olarak gelirse düzleştir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Close kontrolü
    if "Close" not in df.columns:
        raise ValueError(f"{ticker} için 'Close' sütunu bulunamadı.")

    # Close sayısal mı?
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    if df["Close"].isnull().all():
        raise ValueError(f"{ticker} için Close sütunu tamamen boş.")

    return df
