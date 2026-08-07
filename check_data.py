import pandas as pd

df = pd.read_csv('products.csv')
print("Total rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nSample row:\n", df.iloc[0])