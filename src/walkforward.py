# src/walkforward.py

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")


def walkforward_validate(
    df: pd.DataFrame,
    features: list,
    target: str = "Target",
    n_splits: int = 5,
    train_ratio: float = 0.6,
    verbose: bool = True
):
    """
    Walkforward (expanding window) validation.

    Her split'te:
      - Train: baştan train_ratio kadar büyüyen pencere
      - Test:  bir sonraki eşit dilim

    Parametreler
    ------------
    df          : feature + target içeren DataFrame (zaman sırasına göre sıralı)
    features    : model feature listesi
    target      : hedef sütun adı
    n_splits    : kaç fold (varsayılan 5)
    train_ratio : ilk train penceresinin toplam veriye oranı (varsayılan 0.6)
    verbose     : her fold sonucunu yazdır

    Döner
    -----
    results_df  : fold bazlı metrik DataFrame
    """

    df = df.reset_index(drop=True)
    n = len(df)

    # İlk train sonu indeksi
    initial_train_end = int(n * train_ratio)

    # Kalan veriyi n_splits eşit dilimine böl
    remaining = n - initial_train_end
    fold_size = remaining // n_splits

    if fold_size < 30:
        raise ValueError(
            f"Fold başına {fold_size} satır çok az. "
            f"n_splits'i azalt veya daha fazla veri kullan."
        )

    results = []

    for i in range(n_splits):
        train_end = initial_train_end + i * fold_size
        test_start = train_end
        test_end = test_start + fold_size

        if test_end > n:
            break

        X_train = df[features].iloc[:train_end]
        y_train = df[target].iloc[:train_end]
        X_test = df[features].iloc[test_start:test_end]
        y_test = df[target].iloc[test_start:test_end]

        model = XGBClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        buy_rate = y_pred.mean()

        results.append({
            "Fold": i + 1,
            "Train rows": train_end,
            "Test rows": fold_size,
            "F1 (weighted)": round(f1, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Buy signal %": round(buy_rate * 100, 1)
        })

        if verbose:
            print(
                f"  Fold {i+1} | train={train_end} rows | test={fold_size} rows | "
                f"F1={f1:.4f} | Precision={precision:.4f} | Recall={recall:.4f} | "
                f"Buy%={buy_rate*100:.1f}%"
            )

    results_df = pd.DataFrame(results)

    print("\n--- Walkforward Özet ---")
    print(f"  Ortalama F1       : {results_df['F1 (weighted)'].mean():.4f} "
          f"(+/- {results_df['F1 (weighted)'].std():.4f})")
    print(f"  Ortalama Precision: {results_df['Precision'].mean():.4f}")
    print(f"  Ortalama Recall   : {results_df['Recall'].mean():.4f}")
    print(f"  Buy sinyali oranı : %{results_df['Buy signal %'].mean():.1f} (ortalama)")

    return results_df