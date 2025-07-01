import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt


def train_model(df: pd.DataFrame, features, target="Target", model_type="xgb"):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == "xgb":
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        raise ValueError("Desteklenmeyen model türü")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n🧪 Test Sonuçları:")
    print(classification_report(y_test, y_pred, target_names=["Don't Buy", "Buy"]))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Don't Buy", "Buy"],
                yticklabels=["Don't Buy", "Buy"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    sample = X_test.sample(50, random_state=42)
    explain_model_with_shap(model, sample)

    return model

def explain_model_with_shap(model, X_sample):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # SHAP değerlerinin boyutunu kontrol edelim (multiclass mı?)
    print("SHAP shape:", shap_values.values.shape)

    # Eğer çok boyutluysa, sadece "Buy" sınıfı için değerleri al
    if len(shap_values.values.shape) == 3:
        class_idx = 1  # "Buy" sınıfı
        shap_values_for_class = shap.Explanation(
            values=shap_values.values[:, :, class_idx],
            base_values=shap_values.base_values[:, class_idx],
            data=X_sample,
            feature_names=X_sample.columns
        )
    else:
        shap_values_for_class = shap_values  # binary sınıflandırma ise zaten tek boyut

    # Tek bir örnek için waterfall
    shap.plots.waterfall(shap_values_for_class[0])
    plt.show()

    # Beeswarm (özellik etkilerini genel olarak göster)
    shap.plots.beeswarm(shap_values_for_class)
    plt.show()
