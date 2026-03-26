# BenderQuant — Roadmap

Mevcut durum: v2 (walkforward validation + transaction cost modeling)

---

## v3 — Market Regime Detection

**Amaç:** Model her piyasa koşulunda çalışmak yerine, koşula göre sinyal üretip üretmeyeceğine karar versin.

**Yapılacaklar:**
- `src/regime.py` — 3 rejim tanımla:
  - **Trend:** EMA20 > SMA50 ve fiyat SMA50'nin üzerinde
  - **Range:** Bollinger Band genişliği düşük, fiyat band içinde
  - **Volatile:** ATR (Average True Range) son 5 günün ortalamasının üzerinde
- Dataset'e `Regime` feature'ı ekle (0/1/2)
- Model bu feature'ı kullanarak rejime göre farklı ağırlık versin
- Backtest'te volatile rejimde sinyal üretme — sadece trend ve range'de işlem yap

**Beklenti:** MRVL gibi çok trade üreten hisselerde işlem sayısı düşsün, kalite artsın.

---

## v4 — Order Flow Imbalance

**Amaç:** Fiyat hareketinin arkasındaki alıcı/satıcı baskısını ölç.

**Yapılacaklar:**
- `src/order_flow.py` — şu metrikleri hesapla:
  - **OFI (Order Flow Imbalance):** `(Close - Open) / (High - Low)` — günlük bar içi alıcı/satıcı dengesi
  - **Volume Delta:** Yukarı kapanan günlerin volume'u - aşağı kapanan günlerin volume'u (rolling 5 gün)
  - **Buy Pressure:** `(Close - Low) / (High - Low)` — fiyatın günlük range içindeki pozisyonu
- Bu 3 feature'ı `indicators.py`'e ekle
- SHAP ile etkisini ölç — gerçekten sinyal taşıyor mu?

**Beklenti:** SMA50 ve BB dominant olan SHAP sıralamasına yeni feature'lar girsin.

---

## v5 — Rolling Volatility Filter

**Amaç:** Yüksek volatilitede model sinyal üretmesin — en çok zarar bu dönemlerde geliyor.

**Yapılacaklar:**
- `src/vol_filter.py` — ATR hesapla:
  - `ATR = rolling(14).mean(High - Low)`
  - `ATR_Ratio = ATR / Close` — normalize et
  - Eşik: `ATR_Ratio > 0.03` ise volatile → sinyal engelle
- Backtest engine'e `vol_filter=True` parametresi ekle
- Filtreli vs filtresiz backtest karşılaştırması yap

**Beklenti:** Trade sayısı azalsın ama win rate artsın. Sharpe iyileşsin.

---

## v6 — Per-Ticker Walkforward

**Amaç:** Mevcut walkforward 5 hisseyi karıştırıyor. Her hisse için ayrı ayrı çalışsın.

**Yapılacaklar:**
- `walkforward.py`'i güncelle — `per_ticker=True` parametresi ekle
- Her ticker için ayrı expanding window validation
- Sonuçları ticker bazında raporla — hangi hisse gerçekten öğrenilebilir?

**Beklenti:** Bazı hisseler walkforward'da tutarlı çıkacak, bazıları rastgele kalacak. Bu başlı başına önemli bir bulgu.

---

## v7 — Streamlit Dashboard (tamamla)

**Amaç:** `streamlit_app.py` yarım kalmış, bitir ve portfolio'ya taşı.

**Yapılacaklar:**
- Ticker seçimi → canlı veri çek → indikatörleri göster
- Model prediction → Buy/Don't Buy + olasılık skoru
- Backtest sonuçları → equity curve grafiği
- SHAP waterfall plot — "neden bu karar?" açıklaması
- Regime indicator — şu an hangi rejimde?

**Beklenti:** En güçlü portfolio materyali. Görsel, interaktif, açıklanabilir.

---

## Öncelik Sırası

| # | Versiyon | Zorluk | Portfolio Etkisi |
|---|----------|--------|-----------------|
| 1 | v3 — Market Regime | Orta | Yüksek |
| 2 | v4 — Order Flow | Orta | Yüksek |
| 3 | v5 — Vol Filter | Düşük | Orta |
| 4 | v6 — Per-Ticker WF | Düşük | Orta |
| 5 | v7 — Streamlit | Yüksek | Çok Yüksek |

---

## Bilinen Limitasyonlar (değişmeyecek)

- Look-ahead bias: 30 günlük label hâlâ var — düzeltmek için online learning gerekir
- Veri miktarı: 5 hisse × 5 yıl yeterli değil, daha fazla ticker eklemek şart
- Sector bias: AI/semiconductor sektörü 2020-2024 bull market'ta eğitildi
