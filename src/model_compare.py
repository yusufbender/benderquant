import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def compare_models(df: pd.DataFrame, features, target="Target", cv_folds=5):
    X = df[features]
    y = df[target]

    models = {
        "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False, verbosity=0),
        "RandomForest": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LightGBM": LGBMClassifier(verbosity=-1),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print("\n📊 Model Karşılaştırmaları (f1_weighted):")
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"✅ {name:<18} Ort: {scores.mean():.4f}  Std: (+/- {scores.std():.4f})")
