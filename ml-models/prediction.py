import numpy as np
import pandas as pd
import joblib

# load trained model
model = joblib.load('salary_model.pkl')

# define new employee data for prediction
new_data = pd.DataFrame({
    'Experience': [2, 5, 10, 12, 15],
    'Age': [25, 30, 35, 40, 45]
})

# make predictions using trained model
predicted_salaries = model.predict(new_data)

# add predictions to dataframe
new_data['Predicted_Salary'] = predicted_salaries

# print result
print('\n predicted salaries for new employees:')
print(new_data)

# save predictions to csv file
new_data.to_csv('predicted_salaries.csv', index=False)
print('\n predictions saved to predicted_salaries.csv')