import cv2

# Load the image
img = cv2.imread('deep-learning/my_digit.png', cv2.IMREAD_GRAYSCALE)

# Get image shape
print(f"Image Shape: {img.shape}")

# Resize if necessary
if img.shape != (28, 28):
    img = cv2.resize(img, (28, 28))
    cv2.imwrite('deep-learning/my_digit.png', img)
    print("\n✅ Image resized to 28×28 pixels!")