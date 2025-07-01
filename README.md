# 🧠 BenderQuant

**Technical analysis on AI stocks using RSI, EMA, SMA with buy/sell labeling and explainable machine learning (SHAP + Random Forest).**

## 📌 Features

- 📉 Fetches AI-focused stock data using `yfinance`
- 📊 Applies technical indicators: `RSI`, `EMA20`, `SMA50`
- 🎯 Labels data for Buy/Sell using simple forward logic
- 🧪 Trains a `RandomForestClassifier` model
- 🔍 Explains predictions with `SHAP` (Waterfall + Beeswarm)
- 📈 Saves technical indicator plots
- 💾 Outputs datasets & summaries for each stock

## 📈 Tracked AI Stocks

```python
AI_STOCKS = [
    "NVDA",  # Nvidia
    "MRVL",  # Marvell
    "AMD",   # AMD
    "ACLS",  # Axcelis Technologies
    "ON",    # ON Semiconductor
]
```

## ⚙️ Installation

```bash
git clone https://github.com/yusufbender/benderquant.git
cd benderquant
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python main.py
```

## 🧪 Output Example

```
🧪 Test Sonuçları:
              precision    recall  f1-score   support

   Don't Buy       0.86      0.86      0.86        14
         Buy       0.96      0.96      0.96        51

    accuracy                           0.94        65
   macro avg       0.91      0.91      0.91        65
weighted avg       0.94      0.94      0.94        65
```

## ✅ TODO List

- [x] Fetch stock data using `yfinance`
- [x] Add `RSI`, `EMA20`, `SMA50` indicators
- [x] Label data with simple forward-looking strategy
- [x] Train model with Random Forest
- [x] Add SHAP explanations (waterfall + beeswarm)
- [x] Confusion Matrix and Classification Report
- [ ] Add MACD, Bollinger Bands, and Volume indicators
- [ ] Try advanced models: `XGBoost`, `LightGBM`
- [ ] Add web dashboard (Streamlit or Dash)
- [ ] Automate daily pipeline with GitHub Actions
- [ ] Add backtesting module

## 📁 Project Structure

```
benderquant/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── <ticker>_data.csv
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── indicators.py
│   ├── labeling.py
│   ├── summary.py
│   ├── plotter.py
│   ├── model_train.py
│   └── dataset_builder.py
```

## 🧠 Explainable ML (SHAP)

SHAP Waterfall: Visualizes how each feature pushes a prediction.  
SHAP Beeswarm: Shows global feature importance across all predictions.

## 🧑‍💻 Contributing

Pull requests are welcome. For major changes, open an issue first.  
Suggestions, stars, and forks are highly appreciated ⭐

## 📄 License

MIT License – use it, modify it, share it 🙌
