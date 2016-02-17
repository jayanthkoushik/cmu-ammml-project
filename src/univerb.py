# coding=utf-8
# univerb.py: unimodal verb-based classifier.

from __future__ import print_function
import sys
import argparse
import random
from collections import defaultdict

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Dense
from keras.optimizers import Adam
from sklearn.cross_validation import KFold


FEATS_FILE = "data/feats/transc.txt"
EMBEDDING_SIZE = 256
HIDDEN_LAYER_SIZE = 256
DROPOUT_PROB = 0.5
MAX_FEATS = 1000
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 100
VALIDATION = 10
VOCAB_SIZE = 7986

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
args = arg_parser.parse_args()

speaker_data_map = defaultdict(list)
with open(FEATS_FILE) as ff:
    for line in ff:
        d = line.strip().split(" ")
        speaker = d[0]
        y = int(d[1])
        x = map(int, d[2:])
        speaker_data_map[speaker].append((x, y))

print("Building model...", end="")
sys.stdout.flush()
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_SIZE,
                    input_length=MAX_FEATS))
model.add(GRU(output_dim=HIDDEN_LAYER_SIZE, activation="tanh",
              return_sequences=True))
model.add(Dropout(DROPOUT_PROB))
model.add(GRU(output_dim=HIDDEN_LAYER_SIZE, activation="tanh"))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(output_dim=1, activation="sigmoid"))
model.compile(optimizer=Adam(), loss="binary_crossentropy",
              class_mode="binary")
print("done")


def generate_XY(idx, speakers, speaker_data_map):
    X = []
    Y = []
    for i in idx:
        sp = speakers[i]
        for x, y in speaker_data_map[sp]:
            X.append(x)
            Y.append(y)
    Y = np.array(Y)
    return X, Y


speakers = speaker_data_map.keys()
random.shuffle(speakers)
accs = []
for train_idx, test_idx in KFold(len(speakers), VALIDATION):
    X_train, y_train = generate_XY(train_idx, speakers, speaker_data_map)
    X_test, y_test = generate_XY(test_idx, speakers, speaker_data_map)

    X_train = sequence.pad_sequences(X_train, maxlen=MAX_FEATS, padding="post",
                                     truncating="post")
    X_test = sequence.pad_sequences(X_test, maxlen=MAX_FEATS, padding="post",
                                    truncating="post")

    model.fit(X_train, y_train, batch_size=args.batch_size, nb_epoch=args.epochs,
              show_accuracy=True)

    _, acc = model.evaluate(X_test, y_test, batch_size=args.batch_size,
                            show_accuracy=True)
    accs.append(acc)
    print("Test accuracy: {}".format(acc))

final_acc = sum(accs) / VALIDATION
print("Final accuracy: {}".format(final_acc))

