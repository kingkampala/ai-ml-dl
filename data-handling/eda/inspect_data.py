import pandas as pd

# Load the dataset
df = pd.read_csv('dataset.csv')

# Basic info
print("Dataset Info:")
print(df.info())

# Checking for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Checking for duplicates
print("\nDuplicate Rows:\n", df.duplicated().sum())

# Summary statistics
print("\nSummary Statistics:\n", df.describe(include='all'))