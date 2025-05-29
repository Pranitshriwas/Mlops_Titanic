import pandas as pd
import os

# Create raw data folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Load raw data
df = pd.read_csv("tested.csv")

# Save raw data to a consistent location for DVC
df.to_csv("data/raw/train.csv", index=False)

# Basic exploration (optional)
print(df.head())
print(df.columns)
print(df.isnull().sum())
