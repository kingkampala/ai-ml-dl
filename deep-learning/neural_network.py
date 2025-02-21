import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv('cleaned_dataset.csv')

X = df[['Age', 'Experience']]
y = df['Salary']

# scale features (vital 4 deep learning)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# build neural network model
model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(16, activation='relu'),   # input layer
    keras.layers.Dense(8, activation='relu'),     # hidden layer
    keras.layers.Dense(1)     # output layer (1 neuron for salary prediction)
])

# compile model
model.compile(optimizer='adam', loss='mse')

# train model
model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

# evaluate model
loss = model.evaluate(X_test, y_test)
print(f'\n model loss (mse): {loss:.4f}')