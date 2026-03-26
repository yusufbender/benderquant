import os
import pandas as pd
from src.labeling import add_target_label
from src.indicators import add_advanced_features  # 👈 yeni fonksiyon importu

def build_dataset(data_dir="data", features=None):
    if features is None:
        features = [
            "RSI", "EMA20", "SMA50",
            "MACD", "MACD_Signal",
            "BB_Middle", "BB_Upper", "BB_Lower",
            "Volume_Norm",
            "Price_Change_Pct", "RSI_Overbought", "RSI_Oversold",
            "Trend_Crossover", "Volatility", "MACD_Buy_Signal"
        ]

    all_rows = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_data.csv"):
            symbol = filename.replace("_data.csv", "")
            df = pd.read_csv(os.path.join(data_dir, filename), index_col=0)

            if "Close" not in df.columns or df["Close"].dtype == object:
                print(f"⛔ Hatalı dosya: {filename} — Close sütunu yok ya da object tipinde. Atlanıyor.")
                continue

            df["Close"] = df["Close"].astype(float)
            df = add_target_label(df)
            df = add_advanced_features(df)

            # 👇 Symbol sütununu ekle
            df["Symbol"] = symbol.upper()

            df = df[features + ["Target"]].dropna()
            all_rows.append(df)

    dataset = pd.concat(all_rows, ignore_index=True)
    return dataset
