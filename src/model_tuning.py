import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

def run_grid_search(df: pd.DataFrame, features, target="Target"):
    X = df[features]
    y = df[target]

    # Eğitim/Test bölmesi
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Hiperparametre aralığı
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # StratifiedKFold ile daha dengeli çapraz doğrulama
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"\n✅ En iyi parametreler:\n{grid.best_params_}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n🧪 Grid Search Test Sonuçları:")
    print(classification_report(y_test, y_pred, target_names=["Don't Buy", "Buy"]))

    return best_model
