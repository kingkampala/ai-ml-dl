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

model.save('deep-learning/mnist_cnn.h5')
print('\n model saved as mnist_cnn.h5')



'''
import cv2
import os
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

# function to load and preprocess custom handwritten digits
def load_custom_digits(folder):
    custom_images = []
    custom_labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            label = int(filename[0])
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (28, 28))
            img = 255 - img
            img = img / 255
            img = img.reshape(28, 28, 1)
            custom_images.append(img)
            custom_labels.append(label)
    return np.array(custom_images), np.array(custom_labels)

# load custom dataset
custom_folder = 'custom digits'
X_custom, y_custom = load_custom_digits(custom_folder)

# combine mnist and custom data
X_train = np.concatenate((X_train, X_custom), axis=0)
y_train = np.concatenate((y_train, y_custom), axis=0)

# shuffle dataset
shuffle_indices = np.random.permutation(len(X_train))
X_train, y_train = X_train[shuffle_indices], y_train[shuffle_indices]

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
plt.savefig('sample_digits.png')

model.save('deep-learning/custom_mnist_cnn.h5')
print('\n model saved as custom_mnist_cnn.h5')
'''