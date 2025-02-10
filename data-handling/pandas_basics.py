import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'London', 'Paris']
}

df = pd.DataFrame(data)
print("DataFrame:\n", df)

# Basic operations
print("\nSummary Statistics:\n", df.describe())
print("\nSelecting 'Name' column:\n", df['Name'])
print("\nSelecting 'Age' column:\n", df['Age'])

# Filtering data
adults = df[df['Age'] > 28]
print("\nAdults:\n", adults)

# Adding a new column
df['Is_Adult'] = df['Age'] >= 18
print("\nWith 'Is_Adult' Column:\n", df)