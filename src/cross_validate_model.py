import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import numpy as np

def cross_validate_model(df: pd.DataFrame, features, target="Target", cv_folds=5):
    X = df[features]
    y = df[target]

    model = XGBClassifier(eval_metric="logloss", use_label_encoder=False)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted")

    print(f"✅ Stratified {cv_folds}-Fold CV - Ortalama f1_weighted: {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores
