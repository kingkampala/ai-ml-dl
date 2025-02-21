import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load mnist dataset
mnist = keras.datasets.mnist   # load dataset from keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()   # split into train & test

# nomarlize images (scale values btw 0 & 1)
X_train, X_test = X_train / 255.0, X_test / 255.0   # normalize pixel values

# visualize sample images
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.axis('off')
plt.savefig('visualize.png')

# build neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),   # convert 28x28 image to 1D array
    keras.layers.Dense(128, activation='relu'),   # hidden layer (128 neurons)
    keras.layers.Dense(10, activation='softmax')  # output layer (10 classes: 0-9)
])

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'/n test accuracy: {test_acc:.4f}')

# make prediction
predictions = model.predict(X_test)

# show sample prediction
plt.imshow(X_test[0], cmap='gray')
plt.title(f'predicted: {np.argmax(predictions[0])}, actual: {y_test[0]}')
plt.axis('off')
plt.savefig('sample_prediction.png')