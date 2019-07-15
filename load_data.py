import pandas as pd

file = "C:/Users/Magda/Documents/detect-toxic-comments/data/train.csv"
df = pd.read_csv(file)
print(df.head())
print(df.dtypes)