# coding=utf-8
# shallow.py: Shallow 2 layer network.

from keras.models import Sequential
from keras.layers.core import Dense, Dropout


def ShallowNet(input_dim, weights=None):
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    if weights is not None:
        model.load_weights(weights)
    return model
