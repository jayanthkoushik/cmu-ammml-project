# coding=utf-8
# univerb.py: unimodal verb-based classifier.

from __future__ import print_function
import sys
import argparse
import random
import os
from collections import defaultdict

import numpy as np

from datetime import datetime
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Dense
from keras.optimizers import Adam
from sklearn.cross_validation import KFold


FEATS_FILE = "data/feats/transc.txt"
SAVE_PATH = "data/saves/univerb/"
TRAIN_SPLIT_FILE = "data/perssplit/train.txt"
VAL_SPLIT_FILE = "data/perssplit/val.txt"
TEST_SPLIT_FILE = "data/perssplit/test.txt"

EMBEDDING_SIZE = 256
HIDDEN_LAYER_SIZE = 256
DROPOUT_PROB = 0.5
MAX_FEATS = 1000
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 100
VALIDATION = 10
VOCAB_SIZE = 7987
DEFAULT_LEARNING_RATE = 0.0001
GRAD_CLIP = 5

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
arg_parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
args = arg_parser.parse_args()

file_data_map = defaultdict(list)
with open(FEATS_FILE) as ff:
    for line in ff:
        d = line.strip().split(" ")
        file_id = d[0]
        speaker = d[1]
        y = int(d[2])
        x = map(int, d[3:])
        file_data_map[file_id].append((x, y))

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
model.compile(optimizer=Adam(lr=args.lr, clipvalue=GRAD_CLIP),
              loss="binary_crossentropy",
              class_mode="binary")
print("done")

init_weights = model.get_weights()
X_train = []
y_train = []
with open(TRAIN_SPLIT_FILE) as train_split_file:
    for line in train_split_file:
        file_id = line.strip()
        for x, y in file_data_map[file_id]:
            X_train.append(x)
            y_train.append(y)
y_train = np.array(y_train)
X_test = []
y_test = []
with open(VAL_SPLIT_FILE) as val_split_file:
    for line in val_split_file:
        file_id = line.strip()
        for x, y in file_data_map[file_id]:
            X_test.append(x)
            y_test.append(y)
y_test = np.array(y_test)

X_train = sequence.pad_sequences(X_train, maxlen=MAX_FEATS, padding="post",
                                 truncating="post")
X_test = sequence.pad_sequences(X_test, maxlen=MAX_FEATS, padding="post",
                                    truncating="post")
print("No. of samples: train: {}, test: {}".format(X_train.shape[0],
                                                       X_test.shape[0]))

model.set_weights(init_weights)
model.reset_states()
history = model.fit(X_train, y_train, batch_size=args.batch_size,
                    nb_epoch=args.epochs, show_accuracy=True)

_, acc = model.evaluate(X_test, y_test, batch_size=args.batch_size,
                        show_accuracy=True)

print("Validation set test accuracy: {}".format(acc))

fold_save_dir = os.path.join(SAVE_PATH, "{}".format(datetime.now()))
if not os.path.exists(fold_save_dir):
    os.makedirs(fold_save_dir)
model.save_weights(os.path.join(fold_save_dir, "weights.h5"),
                   overwrite=True)
print("\n".join(map(str, history.history["acc"])),
      file=open(os.path.join(fold_save_dir, "accs.txt"), "w"))
print("\n".join(map(str, history.history["loss"])),
      file=open(os.path.join(fold_save_dir, "losses.txt"), "w"))

summary = {
    "learning_rate": args.lr,
    "epochs": args.epochs,
    "batch_size": args.batch_size,
    "final_accuracy": acc
}
print(summary, file=open(os.path.join(fold_save_dir, "summary.txt"), "w"))


