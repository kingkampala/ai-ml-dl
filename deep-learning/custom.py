import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load trained cnn model
model = tf.keras.models.load_model('deep-learning/mnist_cn.h5')

# load and preprocess custom image
img = cv2.imread('deep-learning/my_digit.png', cv2.IMREAD_GRAYSCALE)   # read as grayscale
img = cv2.resize(img, (28, 28))   # resize to 28x28 pixels
img = 255 - img   # invert colours (black background, white digit)
img = img / 255.0   # normalize pixel values (0-1)
img = img.reshape(1, 28, 28, 1)   # reshape as model input

# make a prediction
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)

# show image and prediction
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f'predicted: {predicted_digit}')
plt.axis('off')
plt.savefig('predicted_image.png')

print('\n the model predict this digit as: {predicted_digit}')