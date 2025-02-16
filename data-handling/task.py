# a python script that reads a CSV file and performs basic data cleaning (remove nulls, normalize data).

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# load dataset and print original data
df = pd.read_csv('dataset.csv')

print('original data:\n', df)

# handle missing values
df.fillna({
    'Name': 'Unknown',
    'Age': df['Age'].mean(),
    'Salary': df['Salary'].median()
}, inplace=True)

'''
# covert salary to numeric
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# drop any remaining NaN values in salary
df.dropna(subset=['Salary'], inplace=True)
'''

# normalize numerical data (salary)
scaler = MinMaxScaler()
df[['Salary']] = scaler.fit_transform(df[['Salary']])

# drop duplicate rown (if any)
df.drop_duplicates(inplace=True)

# save cleaned data
df.to_csv('cleaned_dataset.csv', index=False)

print('\nCleaned data:\n', df)
print("\nCleaned dataset saved as 'cleaned_dataset.csv'")