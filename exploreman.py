import pandas as pd

df = pd.read_csv("manifest.csv")
print(df.head())
print(df["label"].value_counts())
print(df["split"].value_counts())
print(df.groupby(["split", "label"]).size())