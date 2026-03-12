import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical

# Load dataset
data = open("dataset.txt").read().lower().split("\n")

# Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

total_words = len(tokenizer.word_index) + 1

input_sequences = []

for line in data:
    token_list = tokenizer.texts_to_sequences([line])[0]
    
    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

max_seq_len = max([len(x) for x in input_sequences])

input_sequences = np.array(
    pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = to_categorical(y, num_classes=total_words)

# Model
model = Sequential()
model.add(Embedding(total_words, 50, input_length=max_seq_len-1))
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(total_words, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, epochs=200, verbose=1)

model.save("next_word_model.h5")

pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

print("Model training completed!")