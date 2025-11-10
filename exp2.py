# Importing modules
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
import matplotlib.pyplot as plt

# Example: loading MNIST dataset (you didn’t include dataset loading)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Cast the records into float values
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize image pixel values by dividing by 255 (gray scale)
x_train /= 255.0
x_test /= 255.0

print("Feature matrix (x_train):", x_train.shape)
print("Target matrix (x_test):", x_test.shape)
print("Feature matrix (y_train):", y_train.shape)
print("Target matrix (y_test):", y_test.shape)

# Display some images
k = 0
plt.figure(figsize=(10, 10))
for i in range(10):
    for j in range(10):
        plt.subplot(10, 10, k + 1)
        plt.imshow(x_train[k], cmap='gray', aspect='auto')
        plt.axis('off')
        k += 1
plt.show()

# Define the model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # flatten 28x28 image into vector
    Dense(256, activation='sigmoid'),  # dense layer 1
    Dense(128, activation='sigmoid'),  # dense layer 2
    Dense(10, activation='sigmoid')    # output layer
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=2000,
    validation_split=0.2
)

# Evaluate the model
results = model.evaluate(x_test, y_test, verbose=0)
print('Test loss, test accuracy:', results)
