import pandas as pd

df = pd.read_parquet("outputs/cleaned/herb.parquet")
print(df.head())
print(df.info())