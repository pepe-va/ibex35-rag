import yfinance as yf

ticker = yf.Ticker("IDR.MC")
price = ticker.fast_info["last_price"]

print(price)