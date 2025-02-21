import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape images for cnn
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# build cnn model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'\n test accuracy: {test_acc:.4f}')

predictions = model.predict(X_test)

plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title(f'predicted: {np.argmax(predictions[0])}, actual: {y_test[0]}')
plt.axis('off')
plt.savefig('sample_20.png')