import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Dataset
data = {
    'Age': [25, 30, 35, 40, 45],
    'Salary': [50000, 60000, 70000, 80000, 90000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# 1. Normalization (Min-Max Scaling)
min_max_scaler = MinMaxScaler()
df['Salary_Normalized'] = min_max_scaler.fit_transform(df[['Salary']])

# 2. Standardization (Z-Score Scaling)
standard_scaler = StandardScaler()
df['Age_Standardized'] = standard_scaler.fit_transform(df[['Age']])

print("\nScaled Data:\n", df)