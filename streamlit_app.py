import streamlit as st
import pandas as pd
import yfinance as yf
import os
import streamlit_authenticator as stauth

from src.model_utils import load_model
from src.indicators import add_indicators, add_advanced_features
from src.backtest import backtest

# 💾 Log klasörü oluştur
os.makedirs("logs", exist_ok=True)

# 🔐 Önceden hashlenmiş şifreler (örnek — sen kendi hasher.py ile üret)
credentials = {
    "usernames": {
        "bender": {
            "name": "Bender",
            "password": "$2b$12$aGZDFU5sUtHxLDajzjIYiuBuo/oj0JkwQNPhxm3kEMKRzZ1oVR2kG",
            "email": "bender@quant.com"
        },
        "guest": {
            "name": "Guest",
            "password": "$2b$12$0dtODdDtWV1bVqaaYfxXA.B/hz4jFduh0m4wMwEEK.PaDhRpvm0He",
            "email": "guest@quant.com"
        }
    }
}

# 🔐 Auth objesi
authenticator = stauth.Authenticate(
    credentials,
    cookie_name="benderquant_cookie",
    key="abcdef123",
    cookie_expiry_days=7
)

# 🔑 Giriş formu
authenticator.login(
    location="main",
    fields={"Form name": "🔐 Giriş Yap", "Username": "Kullanıcı Adı", "Password": "Şifre", "Login": "Giriş"},
    captcha=False
)

# ✅ Kullanıcı girişi kontrolü
auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")

if auth_status:
    st.set_page_config(page_title="📈 BenderQuant", layout="wide")
    st.title(f"🤖 BenderQuant - Hoş geldin, {name}!")

    # Sidebar'da çıkış ve bilgi
    with st.sidebar:
        st.success("✅ Giriş başarılı")
        st.markdown(f"**👤 Kullanıcı:** {name}")
        st.caption(f"📧 {credentials['usernames'][username]['email']}")
        authenticator.logout("🔓 Çıkış Yap", location="sidebar")

    # Özellikler listesi
    FEATURES = [
        "RSI", "EMA20", "SMA50", "MACD", "MACD_Signal", "BB_Middle",
        "BB_Upper", "BB_Lower", "Volume_Norm", "Price_Change_Pct",
        "RSI_Overbought", "RSI_Oversold", "Trend_Crossover",
        "Volatility", "MACD_Buy_Signal"
    ]

    # Kullanıcı inputları
    ticker = st.text_input("🎯 Hisse Sembolü (örnek: NVDA)", "NVDA")
    period = st.selectbox("⏳ Veri Aralığı", ["3mo", "6mo", "1y", "2y"], index=1)

    if st.button("🚀 Tahmin ve Backtest Başlat"):
        model = load_model()
        df = yf.download(ticker, period=period, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = add_indicators(df)
        df = add_advanced_features(df)
        df = df.dropna()

        for col in FEATURES:
            if col not in df.columns:
                df[col] = 0

        df["Prediction"] = model.predict(df[FEATURES])
        df["Signal"] = df["Prediction"].map({1: "📈 AL", 0: "📉 SAT"})

        st.subheader("📊 Tahminler")
        st.dataframe(df[["Close", "Prediction", "Signal"]].tail(30), use_container_width=True)

        st.subheader("💰 Backtest Sonuçları")
        trades_df = backtest(df[["Close", "Prediction"]])
        st.dataframe(trades_df, use_container_width=True)

elif auth_status is False:
    st.error("❌ Hatalı kullanıcı adı veya şifre.")
elif auth_status is None:
    st.warning("👀 Lütfen giriş yap.")
    st.info("🧠 Demo kullanıcı: `bender` / `guest`, şifre: `bender123` / `guest123`")