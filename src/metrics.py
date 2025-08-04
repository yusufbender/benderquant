import numpy as np
import pandas as pd


def calculate_sharpe_ratio(equity_df, risk_free_rate=0.01):
    equity_df = equity_df.copy()
    equity_df["Returns"] = equity_df["Equity"].pct_change()
    excess_returns = equity_df["Returns"] - (risk_free_rate / 252)
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    return round(sharpe_ratio, 3)


def calculate_max_drawdown(equity_df):
    equity = equity_df["Equity"]
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    max_dd = drawdown.min()
    return round(max_dd * 100, 2)  # Yüzde olarak


def calculate_cagr(equity_df):
    start_value = equity_df["Equity"].iloc[0]
    end_value = equity_df["Equity"].iloc[-1]
    days = (equity_df.index[-1] - equity_df.index[0]).days
    if days == 0:
        return 0.0
    cagr = (end_value / start_value) ** (365 / days) - 1
    return round(cagr * 100, 2)


def summarize_performance(equity_df):
    sharpe = calculate_sharpe_ratio(equity_df)
    max_dd = calculate_max_drawdown(equity_df)
    cagr = calculate_cagr(equity_df)
    return {
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "CAGR (%)": cagr
    }