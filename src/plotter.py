import matplotlib.pyplot as plt

def plot_indicators(df, ticker: str):
    plt.figure(figsize=(14, 6))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['EMA20'], label='EMA 20')
    plt.plot(df['SMA50'], label='SMA 50')
    plt.title(f'{ticker} - Price with EMA & SMA')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
