# coding=utf-8
# grutiny.py: tiny GRU classifier.

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dense


def GRU_TINY(vocab_size, embedding_size, max_feats, hidden_layer_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size,
                        input_length=max_feats))
    model.add(GRU(output_dim=hidden_layer_size, activation="tanh"))
    model.add(Dense(output_dim=1, activation="sigmoid"))
    return model
