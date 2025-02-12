import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    'Age': [25, 30, 35, 40, 45],
    'Salary': [50000, 60000, 70000, 80000, 90000],
    'Purchased': [0, 1, 0, 1, 0]  # Target variable
}

df = pd.DataFrame(data)

# Features (X) and Target (y)
X = df[['Age', 'Salary']]  # Independent variables
y = df['Purchased']        # Dependent variable

# Splitting the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output
print("Training Features:\n", X_train)
print("\nTest Features:\n", X_test)
print("\nTraining Target:\n", y_train)
print("\nTest Target:\n", y_test)