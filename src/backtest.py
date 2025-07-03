import pandas as pd

def backtest(df, initial_cash=10000):
    cash = initial_cash
    position = 0
    portfolio = []

    for i in range(len(df)):
        price = df.iloc[i]["Close"]
        signal = df.iloc[i]["Prediction"]

        if signal == 1 and cash > 0:
            position = cash / price
            cash = 0
            portfolio.append(("BUY", df.index[i], price))

        elif signal == 0 and position > 0:
            cash = position * price
            position = 0
            portfolio.append(("SELL", df.index[i], price))

    if position > 0:
        cash = position * df.iloc[-1]["Close"]
        portfolio.append(("FINAL SELL", df.index[-1], df.iloc[-1]["Close"]))

    final_value = cash
    total_return = (final_value - initial_cash) / initial_cash * 100

    print(f"\n💰 Başlangıç: ${initial_cash:.2f}")
    print(f"📈 Bitiş: ${final_value:.2f}")
    print(f"📊 Getiri: %{total_return:.2f}")

    df = pd.DataFrame(portfolio, columns=["Action", "Date", "Price"])
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df
