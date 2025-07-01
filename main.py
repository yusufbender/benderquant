from src.data_loader import get_stock_data
from src.indicators import add_indicators
from src.summary import get_summary_info
from src.plotter import plot_indicators
from src.config import AI_STOCKS
import os
from src.dataset_builder import build_dataset
from src.model_train import train_model




if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

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

            # Sadece ilk hisseyi grafikte gösterelim
            if ticker == AI_STOCKS[0]:
                plot_indicators(df, ticker)

        except Exception as e:
            print(f"❌ {ticker} verisi çekilemedi: {e}")
    dataset = build_dataset()
    print(dataset.head())
    features = ["RSI", "EMA20", "SMA50"]
    model = train_model(dataset, features, model_type="xgb")


