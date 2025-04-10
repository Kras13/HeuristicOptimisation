import yfinance as yf

ticker = "AAPL"

data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

print(data.head())

data.to_csv("apple_stock_data.csv")
