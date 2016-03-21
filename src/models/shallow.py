# coding=utf-8
# shallow.py: Shallow n-layer network.

from keras.models import Sequential
from keras.layers.core import Dense, Dropout


def ShallowNet(input_dim, dropout, dense_layers, dense_layer_units, weights=None):
    model = Sequential()
    model.add(Dropout(dropout, input_shape=(input_dim,)))
    for _ in range(dense_layers):
        model.add(Dense(dense_layer_units, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    if weights is not None:
        model.load_weights(weights)
    return model
