import yfinance as yf, pandas as pd, pathlib
out = pathlib.Path("data/processed")
p = out / "prices.parquet"
df = pd.read_parquet(p)
q = yf.download("QQQ", start="2017-01-01", auto_adjust=True, progress=False)["Close"]
df["QQQ"] = q["QQQ"].reindex(df.index)
df.to_parquet(p)
print("QQQ filled:", int(df["QQQ"].notna().sum()), "of", len(df), "rows")