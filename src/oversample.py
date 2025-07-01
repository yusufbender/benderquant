from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(df: pd.DataFrame, features, target="Target"):
    X = df[features]
    y = df[target]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=features)
    df_resampled[target] = y_resampled

    print(f"🔄 SMOTE uygulandı: {sum(y == 0)} → {sum(y_resampled == 0)}, {sum(y == 1)} → {sum(y_resampled == 1)}")
    return df_resampled
