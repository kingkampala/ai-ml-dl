import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load dataset (use cleaned dataset)
df = pd.read_csv('cleaned_dataset.csv')

# select features (X) and target (y)
X = df[['Age']]    # (independent variable)
y = df['Salary']   # (dependent variable)

# split data into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train linear regression model
model = LinearRegression()   # create model
model.fit(X_train, y_train)  # train model

# make predictions
y_pred = model.predict(X_test)

# visualize results
plt.scatter(X_test, y_test, color='blue', label='Actual Salary')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Age vs Salary Prediction')
plt.legend()
plt.savefig('regression_plot.png')