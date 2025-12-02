import pandas as pd
df = pd.read_csv("../data/nyc_taxi/nyc_taxi.csv")
print(df.shape)
print(df.memory_usage().sum() / 1e6, "MB")