import pandas as pd 
df = pd.read_csv('tested.csv')

print(df.head())

print(df.columns)

print(df.isnull().sum())