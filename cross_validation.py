import pandas as pd
df = pd.read_csv("manifest.csv")
print(df.groupby("label")["duration_s"].describe())