import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional

vocabulary_size = 5000
max_words = 500

print(f"Loading data with vocabulary size = {vocabulary_size}...")
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

print(f"Padding sequences to max length = {max_words}...")
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

print("\n--- Building Model 1: Standard LSTM ---")
embedding_size = 32

model_lstm = Sequential()
model_lstm.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model_lstm.add(LSTM(100))
model_lstm.add(Dense(1, activation='sigmoid'))

print(model_lstm.summary())

model_lstm.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

batch_size = 64
num_epochs = 3

print("\n--- Training Model 1: Standard LSTM ---")
model_lstm.fit(X_train, y_train,
               validation_data=(X_test, y_test),
               batch_size=batch_size,
               epochs=num_epochs,
               verbose=1)

print("\n--- Building Model 2: Bidirectional LSTM (BiLSTM) ---")

model_bilstm = Sequential()
model_bilstm.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model_bilstm.add(Bidirectional(LSTM(100)))  # The only change is wrapping LSTM in Bidirectional
model_bilstm.add(Dense(1, activation='sigmoid'))

print(model_bilstm.summary())

model_bilstm.compile(loss='binary_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])

print("\n--- Training Model 2: BiLSTM ---")
model_bilstm.fit(X_train, y_train,
                 validation_data=(X_test, y_test),
                 batch_size=batch_size,
                 epochs=num_epochs,
                 verbose=1)

print("\n--- Final Results Comparison ---")

scores_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)
print(f"Standard LSTM Test Accuracy: {scores_lstm[1]*100:.2f}%")

scores_bilstm = model_bilstm.evaluate(X_test, y_test, verbose=0)
print(f"Bidirectional LSTM (BiLSTM) Test Accuracy: {scores_bilstm[1]*100:.2f}%")
