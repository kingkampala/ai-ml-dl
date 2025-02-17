import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# load cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

df = df.drop(columns=['Name'], errors='ignore')

df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# select features (X) and target (y)
X = df.drop(columns=['Salary'])    # independent variable
y = df['Salary']   # dependent variable

'''
salary_std = np.std(df['Salary'])
salary_mean = np.mean(df['Salary'])

print(f"Salary Standard Deviation: {salary_std:.2f}")
print(f"Salary Mean: {salary_mean:.2f}")

sns.pairplot(df[['Age', 'Experience', 'Salary']])
plt.savefig('show.png')

# Scatter plot of Age vs Salary
sns.scatterplot(x=df['Age'], y=df['Salary'])
plt.title("Age vs Salary")
plt.savefig('one.png')

# Scatter plot of Experience vs Salary
sns.scatterplot(x=df['Experience'], y=df['Salary'])
plt.title("Experience vs Salary")
plt.savefig('two.png')

print(df[['Age', 'Salary']].corr())
print(df[['Experience', 'Salary']].corr())
'''

#df.dropna(subset=['Age', 'Salary'], inplace=True)

# split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print results
print(f'model evaluation results:')
print(f'mean squared error (mse): {mse:.2f}')
print(f'mean absolute error (mae): {mae:.2f}')
print(f'r² score: {r2:.2f}')

# save trained model
joblib.dump(model, 'optimized_salary_model.pkl')
print('\n optimized model saved as optimized_salary_model.pkl')