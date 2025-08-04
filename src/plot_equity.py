import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curve(equity_df, trades_df=None, title="Portföy Büyüklüğü (Equity Curve)", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    equity_df.plot(ax=ax, legend=False)
    ax.set_title(title)
    ax.set_ylabel("Equity ($)")
    ax.set_xlabel("Tarih")
    ax.grid(True, linestyle="--", alpha=0.3)

    # BUY / SELL noktaları
    if trades_df is not None:
        buys = trades_df[trades_df["Action"] == "BUY"]
        sells = trades_df[trades_df["Action"].str.contains("SELL")]

        for _, row in buys.iterrows():
            ax.axvline(row.name, color="green", linestyle=":", alpha=0.6)
            ax.scatter(row.name, row["Price"], marker="^", color="green")

        for _, row in sells.iterrows():
            ax.axvline(row.name, color="red", linestyle=":", alpha=0.6)
            ax.scatter(row.name, row["Price"], marker="v", color="red")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"📸 Grafik kaydedildi: {save_path}")
    else:
        plt.show()

    plt.close()
