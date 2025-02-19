import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('cleaned_dataset.csv')

df = df.drop(columns=['Name'], errors='ignore')

df = pd.get_dummies(df, columns=['Department'], drop_first=True)

X = df.drop(columns=['Salary'])
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=10)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MSE': mse, 'MAE': mae, 'R²': r2}

print('\n model comparison results:')
for name, metrics in  results.items():
    print('\n{name}:')
    print(f' MSE: {metrics['MSE']:.2f}')
    print(f' MAE: {metrics['MAE']:.2f}')
    print(f' R² Score: {metrics['R²']:.2f}')