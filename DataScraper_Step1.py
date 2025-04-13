import yfinance as yf

data = yf.download("AAPL", start="2010-01-01", end="2025-01-01")

print(data.head())
data.to_csv("original_data.csv")