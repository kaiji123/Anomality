import yfinance as yf, pandas as pd, pathlib
TICKERS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","NFLX","INTC",
"ORCL","CRM","ADBE","CSCO","QCOM","JPM","BAC","GS","MS","WFC","V","MA","BRK-B",
"UNH","JNJ","PFE","MRK","LLY","ABBV","XOM","CVX","COP","WMT","COST","HD","MCD",
"NKE","KO","PEP","DIS","BA","CAT","GE","UPS","T","SPY","QQQ","IWM",
"BTC-USD","ETH-USD","SOL-USD","XRP-USD","DOGE-USD","ADA-USD","AVAX-USD","LINK-USD"]
out = pathlib.Path("data/processed"); out.mkdir(parents=True, exist_ok=True)
df = yf.download(TICKERS, start="2017-01-01", auto_adjust=True, progress=False)["Close"]
df.to_parquet(out / "prices.parquet")
print("wrote", df.shape, "to", out / "prices.parquet")