from src.multi_backtest import multi_backtest

tickers = ["NVDA", "AMD", "MRVL", "ACLS", "ON"]
summary_df = multi_backtest(tickers)

print("\n📊 Çoklu Hisse Backtest Özeti:")
print(summary_df)
