from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib

# load cleaned dataset
df = pd.read_csv('cleaned_dataset.csv')

df = df.drop(columns=['Name'], errors='ignore')

# one-hot encode 'Department'
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# select features and target
X = df.drop(columns=['Salary'])
y = df['Salary']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# define model
model = RandomForestRegressor()

# define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# perform grid search
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# get the best model
best_model = grid_search.best_estimator_

# make predictions
y_pred = best_model.predict(X_test)

# evaluate model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# print results
print(f'\n model evaluation after hyperparameter tuning:')
print(f'mean squared error (mse): {mse:.2f}')
print(f'mean absolute error (mae): {mae:.2f}')
print(f'r² score: {r2:.2f}')

# save the best_model
joblib.dump(best_model, 'best_salary_model.pkl')
print('\n best optimized model saved as best_salary_model.pkl')    