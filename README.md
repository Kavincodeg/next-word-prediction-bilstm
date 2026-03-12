# Next Word Prediction using Bidirectional LSTM

This project predicts the next word in a sentence using a deep learning model built with TensorFlow and Keras.

## Project Description

The model learns patterns from a text dataset and predicts the most probable next word given an input sentence.

## Technologies Used

* Python
* TensorFlow / Keras
* Bidirectional LSTM
* Natural Language Processing (NLP)

## Model Architecture

Embedding Layer
↓
Bidirectional LSTM
↓
Bidirectional LSTM
↓
Dense Softmax Layer

## Example

Input:
i will be

Output:
soon

## How to Run

Train the model:

python train_model.py

Predict next word:

python predict.py
