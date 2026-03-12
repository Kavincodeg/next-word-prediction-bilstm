import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model and tokenizer
model = load_model("next_word_model.h5")
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

max_seq_len = 5   # sequence length used during training

def predict_next_word(text):
    
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

    predicted = np.argmax(model.predict(token_list), axis=-1)

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            return word

# Input
text = input("Enter a sentence: ")

next_word = predict_next_word(text)

print("Predicted next word:", next_word)