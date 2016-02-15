# coding=utf-8
# univerb.py: unimodal verb-based classifier.

from __future__ import print_function
import sys

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Dense
from keras.optimizers import Adam


VOCAB_FILE = "data/vocab.txt"
FEATS_FILE = "data/feats/transc.txt"
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 128
DROPOUT_PROB = 0.5
BATCH_SIZE = 64
MAX_FEATS = 1000
TRAIN_SAMPLES = 200
EPOCHS = 1

X = []
y = []
with open(FEATS_FILE) as ff:
    for line in ff:
        d = map(int, line.strip().split(" "))
        y.append(d[0])
        X.append(d[1:])
X = sequence.pad_sequences(X, padding="post", truncating="post",
                           maxlen=MAX_FEATS)
y = np.array(y)

# Split the data into training and test.
indices = np.random.permutation(X.shape[0])
train_idx, test_idx = indices[:TRAIN_SAMPLES], indices[TRAIN_SAMPLES:]
X_train, X_test = X[train_idx, :], X[test_idx, :]
y_train, y_test = y[train_idx], y[test_idx]

with open(VOCAB_FILE) as vf:
    # 1 extra for unknown words, and 1 for the pad word.
    vocab_size = 2 + sum(1 for line in vf)

print("Building model...", end="")
sys.stdout.flush()
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_SIZE,
                    input_length=MAX_FEATS))
model.add(GRU(output_dim=HIDDEN_LAYER_SIZE, activation="tanh"))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(output_dim=1, activation="sigmoid"))

model.compile(optimizer=Adam(), loss="binary_crossentropy",
              class_mode="binary")
print("done")

model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS,
          validation_data=(X_test, y_test), show_accuracy=True)

_, acc = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE,
                        show_accuracy=True)
print("Test accuracy: {}".format(acc))

