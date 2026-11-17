import pandas as pd
import gzip

#yjmob100k-dataset1.csv.gz

import pandas as pd

df = pd.read_csv("yjmob100k-dataset1.csv")

print(df.head())
print(df.info())
print(df.describe())