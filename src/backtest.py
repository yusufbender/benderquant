import pandas as pd


def backtest(
    df,
    initial_cash=10000,
    stop_loss=0.05,
    take_profit=0.1,
    verbose=True
):
    cash = initial_cash
    position = 0
    entry_price = 0
    portfolio_log = []
    portfolio_values = []

    for i in range(len(df)):
        date = df.index[i]
        price = df.iloc[i]["Close"]
        signal = df.iloc[i]["Prediction"]

        # Günlük portföy değeri hesapla
        current_value = cash + position * price
        portfolio_values.append((date, current_value))

        # Satın alma
        if signal == 1 and cash > 0:
            position = cash / price
            entry_price = price
            cash = 0
            portfolio_log.append(("BUY", date, price))
            if verbose:
                print(f"BUY @{price:.2f} on {date}")

        # Stop-loss veya take-profit
        elif position > 0:
            change = (price - entry_price) / entry_price

            if signal == 0 or change <= -stop_loss or change >= take_profit:
                cash = position * price
                position = 0
                portfolio_log.append(("SELL", date, price))
                if verbose:
                    print(f"SELL @{price:.2f} on {date} | Return: {change*100:.2f}%")

    # Pozisyon açık kaldıysa son gün kapat
    if position > 0:
        cash = position * df.iloc[-1]["Close"]
        portfolio_log.append(("FINAL SELL", df.index[-1], df.iloc[-1]["Close"]))

    final_value = cash
    total_return = (final_value - initial_cash) / initial_cash * 100

    if verbose:
        print(f"\n💰 Başlangıç: ${initial_cash:.2f}")
        print(f"📈 Bitiş: ${final_value:.2f}")
        print(f"📊 Getiri: %{total_return:.2f}")

    trades_df = pd.DataFrame(portfolio_log, columns=["Action", "Date", "Price"])
    trades_df["Date"] = pd.to_datetime(trades_df["Date"])
    trades_df.set_index("Date", inplace=True)

    equity_df = pd.DataFrame(portfolio_values, columns=["Date", "Equity"])
    equity_df.set_index("Date", inplace=True)

    return trades_df, equity_df
