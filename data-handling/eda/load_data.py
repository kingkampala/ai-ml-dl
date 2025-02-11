import pandas as pd
import numpy as np

np.random.seed(42)
names = ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Irene', 'John']
departments = ['HR', 'Finance', 'IT', 'Marketing']

# Simulating a CSV dataset (you can replace this with an actual CSV file)
data = {
    '''
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [25, 30, 35, None, 40],
    'Salary': [50000, 60000, None, 80000, 90000],
    'Department': ['HR', 'Finance', 'IT', 'HR', None]
    '''
    'Name': np.random.choice(names, 50),
    'Age': np.random.randint(22, 50, 50),
    'Salary': np.random.randint(40000, 120000, 50),
    'Department': np.random.choice(departments, 50),
    'Experience': np.random.randint(1, 15, 50)
}

# Saving the simulated data as a CSV file
df = pd.DataFrame(data)
df.to_csv('dataset.csv', index=False)

# Loading the dataset
df_loaded = pd.read_csv('dataset.csv')
print("Loaded Dataset:\n", df_loaded)