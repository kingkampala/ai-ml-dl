import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess MNIST dataset
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Function to load and preprocess custom handwritten digits
def load_custom_digits(folder):
    custom_images = []
    custom_labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            label = int(filename[0])  # Assumes filenames are like "8_digit.png"
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = 255 - img  # Invert colors (if needed)
            img = img / 255.0  # Normalize
            img = img.reshape(28, 28, 1)
            custom_images.append(img)
            custom_labels.append(label)
    return np.array(custom_images), np.array(custom_labels)

# Load custom dataset
custom_folder = "custom digits"  # Change this to your folder path
X_custom, y_custom = load_custom_digits(custom_folder)

# Combine MNIST and custom data
X_train = np.concatenate((X_train, X_custom), axis=0)
y_train = np.concatenate((y_train, y_custom), axis=0)

# Shuffle the dataset
shuffle_indices = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]

# Build CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\nTest accuracy: {test_acc:.4f}')

# Save model
model.save('deep-learning/mnist_custom_cnn.h5')
print('\nModel saved as mnist_custom_cnn.h5')
