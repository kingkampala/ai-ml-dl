import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# load trained model
model = tf.keras.models.load_model('deep-learning/mnist_cnn.h5')

# load image
image = cv2.imread('deep-learning/my_digit.png', cv2.IMREAD_GRAYSCALE)

# apply threshold to make digits standout
_, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

# find contours (detect digits)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# sort contours from left to right
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

predictions = []

# loop through each detected digit
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    digit = thresh[y:y+h, x:x+w]   # crop the digit

    # resize to 28x28 (mnist format)
    digit = cv2.resize(digit, (28, 28))

    # normalize (convert pixel values from 0-255 to 0-1)
    digit = digit.astype('float32') / 255.0

    # reshape for model input
    digit = np.expand_dims(digit, axis=0)   # add batch dimension
    digit = np.expand_dims(digit, axis=-1)   # add channel dimension

    # predict digit
    prediction = np.argmax(model.predict(digit), axis=-1)[0]
    predictions.append(prediction)

# display results
print('predicted digits:', predictions)
plt.imshow(thresh, cmap='gray')
plt.title(f'predicted: {predictions}')
plt.savefig('predicted_digits.png')