# 🧠 BenderQuant

**AI hisseleri üzerinde teknik analiz yapan, SHAP ile açıklanabilir makine öğrenimi kullanan çok modellerli bir sistem.**

## 📌 Özellikler

- 📉 AI şirketlerinin borsa verisini `yfinance` ile otomatik çeker  
- 📊 Teknik göstergeleri hesaplar: `RSI`, `EMA20`, `SMA50`, `MACD`, `Bollinger Bands`, `Volume`
- 🎯 Basit ileriye dönük etiketleme ile `Buy` / `Don't Buy` sınıflandırması
- 🤖 Çoklu model desteği: `Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`, `Logistic Regression`
- ⚖️ SMOTE ile veri dengesizliği çözümü
- 🔍 SHAP ile model açıklamaları: `waterfall`, `beeswarm`
- 🧪 GridSearchCV ve Stratified K-Fold ile hiperparametre optimizasyonu
- 📈 Performans karşılaştırması (F1, Precision, Recall)
- 🧬 Özellik önem sıralamaları + görselleştirme
- 📁 Verileri `data/` klasörüne kaydeder, `summary.txt` üretir
- 💡 Her hisse için görsel teknik analiz grafiği oluşturur

## 🧠 Takip Edilen AI Hisseleri

```python
AI_STOCKS = [
    "NVDA",  # Nvidia
    "MRVL",  # Marvell
    "AMD",   # AMD
    "ACLS",  # Axcelis Technologies
    "ON",    # ON Semiconductor
]
```

## ⚙️ Kurulum

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

## ✅ Çıktı Örnekleri

### 🎯 Sınıflandırma Raporu

```
🧪 Test Sonuçları:
              precision    recall  f1-score   support
   Don't Buy       0.86      0.86      0.86        14
         Buy       0.96      0.96      0.96        51
    accuracy                           0.94        65
```

### 📊 Model Karşılaştırması (SMOTE sonrası)

| Model             | f1_weighted | Std     |
|------------------|-------------|---------|
| ✅ LightGBM       | 0.9799      | ±0.0250 |
| ✅ RandomForest   | 0.9741      | ±0.0264 |
| ✅ CatBoost       | 0.9741      | ±0.0278 |
| 🟨 XGBoost        | 0.9683      | ±0.0291 |
| ⚠️ LogisticReg.  | 0.7238      | ±0.0364 |

## ✅ Yapılacaklar

- [x] RSI, EMA, MACD, BB, Volume gibi teknik göstergeler
- [x] Buy/Sell etiketleme
- [x] XGBoost, LightGBM, CatBoost ile model karşılaştırması
- [x] GridSearchCV ile hiperparametre tuning
- [x] SMOTE ile veri dengesi
- [x] SHAP ile açıklanabilirlik
- [ ] 🔜 Streamlit veya Dash ile görsel web arayüzü
- [ ] 🔜 Günlük otomasyon için GitHub Actions
- [ ] 🔜 Backtesting modülü (geçmişe dönük test)

## 📁 Proje Yapısı

```
benderquant/
├── main.py
├── requirements.txt
├── README.md
├── data/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── indicators.py
│   ├── summary.py
│   ├── plotter.py
│   ├── dataset_builder.py
│   ├── model_train.py
│   ├── model_tuning.py
│   ├── cross_validate_model.py
│   ├── oversample.py
│   └── model_compare.py
```

## 🔍 SHAP Açıklamaları

- **SHAP Waterfall**: Bir örnek için kararın neden Buy/Don't Buy olduğunu detaylandırır.
- **SHAP Beeswarm**: Tüm verilerde hangi özelliklerin daha etkili olduğunu görselleştirir.

## 📄 Lisans

MIT Lisansı — kullan, geliştir, paylaş 🚀

---

> Bu proje bir yatırım tavsiyesi değildir. Eğitimsel amaçlarla geliştirilmiştir.
