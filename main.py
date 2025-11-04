import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def main():
    print("Hello World")
    ticker_name = "AAPL"
    data = yf.download(ticker_name, period="1mo", auto_adjust=True)
    prices = []
    dates = []
    for item in data.get("Close").get(ticker_name):
        prices.append(item)

    dates = [item for item in data.index]

    sma = calculateSMA(prices)

    plt.plot(dates, prices)
    plt.xlabel("Date")
    plt.ylabel("Price")

    ax = plt.gca()
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.title(f"Stock Prices | SMA = {sma}")

    plt.show()


def calculateSMA(prices):
    length = len(prices)

    return sum(prices) / length

if __name__ == "__main__":
    main()