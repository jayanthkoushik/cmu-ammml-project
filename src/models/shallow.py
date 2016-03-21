# coding=utf-8
# shallow.py: Shallow 2 layer network.

from keras.models import Sequential
from keras.layers.core import Dense, Dropout


def ShallowNet(input_shape, weights=None):
    model = Sequential()
    model.add(Dense(100, input_shape=input_shape, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    if weights is not None:
        model.load_weights(weights)
    return model
