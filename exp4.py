from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import (
    Dense, Input, Flatten,
    Reshape, LeakyReLU as LR,
    Activation, Dropout
)
from tensorflow.keras.models import Model, Sequential
from matplotlib import pyplot as plt
from IPython import display  
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

plt.imshow(x_train[0], cmap="gray")
plt.title("Sample Image from MNIST")
plt.show()

LATENT_SIZE = 32

encoder = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512), LR(),
    Dropout(0.5),
    Dense(256), LR(),
    Dropout(0.5),
    Dense(128), LR(),
    Dropout(0.5),
    Dense(64), LR(),
    Dropout(0.5),
    Dense(LATENT_SIZE), LR()
])

decoder = Sequential([
    Dense(64, input_shape=(LATENT_SIZE,)), LR(),
    Dropout(0.5),
    Dense(128), LR(),
    Dropout(0.5),
    Dense(256), LR(),
    Dropout(0.5),
    Dense(512), LR(),
    Dropout(0.5),
    Dense(784),
    Activation("sigmoid"),
    Reshape((28, 28))
])

img = Input(shape=(28, 28))
latent_vector = encoder(img)
output = decoder(latent_vector)

model = Model(inputs=img, outputs=output)
model.compile(optimizer="nadam", loss="binary_crossentropy")

model.summary()

EPOCHS = 5
BATCH_SIZE = 128

history = model.fit(
    x_train, x_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(x_test, x_test),
    verbose=1
)

for epoch in range(5):  # Show only 5 iterations for demo
    fig, axs = plt.subplots(4, 4, figsize=(5, 5))
    rand = x_test[np.random.randint(0, 10000, 16)]
    
    display.clear_output(wait=True)
    
    for i in range(4):
        for j in range(4):
            reconstructed = model.predict(rand[i * 4 + j].reshape(1, 28, 28))
            axs[i, j].imshow(reconstructed[0], cmap="gray")
            axs[i, j].axis("off")
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    print("----------- EPOCH", epoch + 1, "-----------")
