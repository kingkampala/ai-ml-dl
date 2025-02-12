import pandas as pd
import numpy as np

# Simulated dataset with missing values
data = {
    'Name': ['Alice', 'Bob', np.nan, 'David', 'Eva'],
    'Age': [25, np.nan, 30, 40, np.nan],
    'Salary': [50000, 60000, np.nan, 80000, 90000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# 1. Fill missing 'Name' with 'Unknown'
df['Name'].fillna('Unknown', inplace=True)

# 2. Fill missing 'Age' with the mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# 3. Drop rows where 'Salary' is missing
df.dropna(subset=['Salary'], inplace=True)

print("\nCleaned Data:\n", df)