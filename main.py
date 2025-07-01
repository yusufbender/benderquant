from src.data_loader import get_stock_data
from src.indicators import add_indicators
from src.summary import get_summary_info
from src.plotter import plot_indicators
from src.config import AI_STOCKS
from src.dataset_builder import build_dataset
from src.model_train import train_model
from src.model_tuning import run_grid_search
from src.cross_validate_model import cross_validate_model
from src.oversample import apply_smote
from src.model_compare import compare_models

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*use_label_encoder.*")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # 🔁 Veri çek ve ön işleme
    for ticker in AI_STOCKS:
        print(f"\n🔄 İşleniyor: {ticker}")
        try:
            df = get_stock_data(ticker)
            df = add_indicators(df)
            df.to_csv(os.path.join("data", f"{ticker}_data.csv"), index=True)

            info = get_summary_info(ticker)
            with open(os.path.join("data", f"{ticker}_summary.txt"), "w", encoding="utf-8") as f:
                for key, value in info.items():
                    f.write(f"{key}: {value}\n")

            if ticker == AI_STOCKS[0]:
                plot_indicators(df, ticker)

        except Exception as e:
            print(f"❌ {ticker} verisi çekilemedi: {e}")

    # 📦 Dataset oluştur
    dataset = build_dataset()
    print(dataset.head())

    # 🔎 Özellikleri belirle
    features = dataset.columns.drop("Target").tolist()

    # 🎯 Model eğitimi (XGBoost + SHAP)
    train_model(dataset, features, model_type="xgb")

    # 🔧 GridSearchCV ile hiperparametre optimizasyonu
    run_grid_search(dataset, features)

    # 🧪 Cross-validation değerlendirmesi
    cross_validate_model(dataset, features, cv_folds=5)

    # 📉 SMOTE ÖNCESİ model karşılaştırması (baseline için)
    print("\n📉 SMOTE ÖNCESİ SONUÇLAR")
    compare_models(dataset, features)

    # 📈 SMOTE SONRASI model karşılaştırması
    print("\n📈 SMOTE SONRASI SONUÇLAR")
    smote_dataset = apply_smote(dataset, features)
    compare_models(smote_dataset, features)
