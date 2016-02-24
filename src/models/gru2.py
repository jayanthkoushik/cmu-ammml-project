# coding=utf-8
# gru2.py: 2 layer GRU RNN binary classifier.

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Dense


def GRU2(vocab_size, embedding_size, max_feats, hidden_layer_size,
        dropout_prob):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size,
                        input_length=max_feats))
    model.add(GRU(output_dim=hidden_layer_size, activation="tanh",
                return_sequences=True))
    model.add(Dropout(dropout_prob))
    model.add(GRU(output_dim=hidden_layer_size, activation="tanh"))
    model.add(Dropout(dropout_prob))
    model.add(Dense(output_dim=1, activation="sigmoid"))
    return model

