import yfinance as yf

def get_summary_info(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "Symbol": info.get("symbol"),
        "Name": info.get("shortName"),
        "Sector": info.get("sector"),
        "Market Cap": info.get("marketCap"),
        "P/E Ratio": info.get("trailingPE"),
        "EPS": info.get("trailingEps"),
        "52 Week High": info.get("fiftyTwoWeekHigh"),
        "52 Week Low": info.get("fiftyTwoWeekLow"),
        "AI Mention (manual)": "Yes" if "AI" in info.get("longBusinessSummary", "") else "No"
    }
