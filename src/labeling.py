import pandas as pd

def add_target_label(df, threshold=0.05, horizon=30):
    """
    Verilen DataFrame'e hedef etiketini ekler.
    Eğer kapanış fiyatı 30 gün sonra %5 artmışsa: 1, değilse: 0
    """
    df = df.copy()
    df["Close"] = df["Close"].astype(float)
    df["Future_Close"] = df["Close"].shift(-horizon)
    df["Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    df["Target"] = (df["Return"] > threshold).astype(int)
    df.dropna(inplace=True)
    return df
