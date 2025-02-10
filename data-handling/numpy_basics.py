import numpy as np

# Create a NumPy array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

# Basic operations
print("Mean:", np.mean(arr))
print("Standard Deviation:", np.std(arr))
print("Sum:", np.sum(arr))

# 2D Array (Matrix)
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print("\nMatrix:\n", matrix)

# Transposing a matrix
print("Transposed Matrix:\n", matrix.T)

# Element-wise operations
squared = arr ** 2
print("Squared Elements:", squared)