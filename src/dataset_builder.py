import os
import pandas as pd
from src.labeling import add_target_label

def build_dataset(data_dir="data", features=None):
    if features is None:
        features = ["RSI", "EMA20", "SMA50"]

    all_rows = []
    for filename in os.listdir(data_dir):
        if filename.endswith("_data.csv"):
            df = pd.read_csv(os.path.join(data_dir, filename), index_col=0)

            if "Close" not in df.columns or df["Close"].dtype == object:
                print(f"⛔ Hatalı dosya: {filename} — Close sütunu yok ya da object tipinde. Atlanıyor.")
                continue

            df["Close"] = df["Close"].astype(float)
            df = add_target_label(df)
            df = df[features + ["Target"]].dropna()
            all_rows.append(df)

    dataset = pd.concat(all_rows, ignore_index=True)
    return dataset
