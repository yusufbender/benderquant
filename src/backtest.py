# src/backtest.py

import pandas as pd


def backtest(
    df,
    initial_cash=10000,
    stop_loss=0.05,
    take_profit=0.1,
    commission=0.001,
    slippage=0.0005,
    verbose=True
):
    """
    Backtest engine with transaction cost modeling.

    Parametreler
    ------------
    df          : Close + Prediction sütunları olan DataFrame
    initial_cash: Başlangıç sermayesi (varsayılan $10,000)
    stop_loss   : Stop-loss eşiği (varsayılan %5)
    take_profit : Take-profit eşiği (varsayılan %10)
    commission  : Her işlemde ödenen komisyon oranı (varsayılan %0.1)
    slippage    : Her işlemde gerçekleşen slippage oranı (varsayılan %0.05)
                  Alımda fiyatı artırır, satışta fiyatı düşürür.
    verbose     : İşlem loglarını yazdır
    """

    cash = initial_cash
    position = 0
    entry_price = 0
    total_cost = 0
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
            buy_price = price * (1 + slippage)           # slippage: biraz daha pahalıya al
            cost = cash * commission                      # komisyon
            total_cost += cost
            cash -= cost
            position = cash / buy_price
            entry_price = buy_price
            cash = 0
            portfolio_log.append(("BUY", date, price, buy_price, cost))
            if verbose:
                print(f"BUY  @{price:.2f} (eff: {buy_price:.2f}) | commission: ${cost:.2f} | {date}")

        # Stop-loss veya take-profit veya sat sinyali
        elif position > 0:
            change = (price - entry_price) / entry_price

            if signal == 0 or change <= -stop_loss or change >= take_profit:
                sell_price = price * (1 - slippage)      # slippage: biraz daha ucuza sat
                gross = position * sell_price
                cost = gross * commission                 # komisyon
                total_cost += cost
                cash = gross - cost
                position = 0
                portfolio_log.append(("SELL", date, price, sell_price, cost))
                if verbose:
                    print(
                        f"SELL @{price:.2f} (eff: {sell_price:.2f}) | "
                        f"return: {change*100:.2f}% | commission: ${cost:.2f} | {date}"
                    )

    # Pozisyon açık kaldıysa son gün kapat
    if position > 0:
        last_price = df.iloc[-1]["Close"]
        sell_price = last_price * (1 - slippage)
        gross = position * sell_price
        cost = gross * commission
        total_cost += cost
        cash = gross - cost
        portfolio_log.append(("FINAL SELL", df.index[-1], last_price, sell_price, cost))

    final_value = cash
    total_return = (final_value - initial_cash) / initial_cash * 100

    if verbose:
        print(f"\n💰 Başlangıç : ${initial_cash:.2f}")
        print(f"📈 Bitiş     : ${final_value:.2f}")
        print(f"💸 Toplam maliyet (komisyon + slippage): ${total_cost:.2f}")
        print(f"📊 Net getiri: %{total_return:.2f}")

    trades_df = pd.DataFrame(
        portfolio_log,
        columns=["Action", "Date", "Price", "Effective Price", "Cost"]
    )
    trades_df["Date"] = pd.to_datetime(trades_df["Date"])
    trades_df.set_index("Date", inplace=True)

    equity_df = pd.DataFrame(portfolio_values, columns=["Date", "Equity"])
    equity_df.set_index("Date", inplace=True)

    return trades_df, equity_df
