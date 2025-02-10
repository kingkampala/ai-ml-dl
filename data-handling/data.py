import pandas as pd
import numpy as np

# Simulated messy data
data = {
    'Name': ['Alice', 'Bob', np.nan, 'David'],
    'Age': [25, np.nan, 30, 40],
    'Salary': [50000, 60000, np.nan, 80000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Handling missing values
df['Name'].fillna('Unknown', inplace=True)        # Replace NaN in 'Name' with 'Unknown'
df['Age'].fillna(df['Age'].mean(), inplace=True)  # Replace NaN in 'Age' with mean
df.dropna(subset=['Salary'], inplace=True)        # Drop rows where 'Salary' is NaN

# Normalizing the Salary column
df['Normalized_Salary'] = (df['Salary'] - df['Salary'].min()) / (df['Salary'].max() - df['Salary'].min())

print("\nCleaned Data:\n", df)