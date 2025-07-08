# BenderQuant

Borsa ve yapay zeka şirketleri için, farklı makine öğrenimi modelleriyle teknik analiz yapan ve sonuçlarını görselleştiren, açıklamalarıyla şeffaf bir Python projesi.

## Özellikler

- Otomatik veri çekme (`yfinance`)
- Teknik göstergeler: RSI, EMA20, SMA50, MACD, Bollinger Bands, Volume
- Basit “Al / Alma” sınıflandırma
- Farklı model seçenekleri (Random Forest, XGBoost, LightGBM, CatBoost, Lojistik Regresyon)
- SMOTE ile veri dengesi iyileştirmesi
- SHAP ile model kararlarının görsel açıklaması
- Hiperparametre optimizasyonu (GridSearchCV)
- Performans metriklerinin karşılaştırılması
- Özellik önem sıralaması ve grafikler
- Her hisse için teknik analiz grafiği
- Data klasöründe çıktı dosyası ve özet rapor

## İzlenen Hisseler

NVDA, MRVL, AMD, ACLS, ON gibi bazı AI ve yarı iletken firmaları.

## Kurulum
bash

```
git clone https://github.com/yusufbender/benderquant.git
cd benderquant
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python main.py
