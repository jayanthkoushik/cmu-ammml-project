# coding=utf-8
# univerb.py: unimodal verb-based classifier.

from __future__ import print_function
import sys
import argparse
import random
import os
from collections import defaultdict

import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.core import Dropout, Dense
from keras.optimizers import Adam
from sklearn.cross_validation import KFold
from gensim.models import Word2Vec


FEATS_FILE = "data/feats/transc.txt"
SAVE_PATH = "data/saves/univerb/"
HIDDEN_LAYER_SIZE = 256
DROPOUT_PROB = 0.5
MAX_FEATS = 1000
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 100
VALIDATION = 2
LEARNING_RATE = 0.0001
GRAD_CLIP = 5

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
arg_parser.add_argument("--embedding-file", type=str, required=True)
args = arg_parser.parse_args()

speaker_data_map = defaultdict(list)
with open(FEATS_FILE) as ff:
    for line in ff:
        d = line.strip().split(" ")
        speaker = d[0]
        y = int(d[1])
        x = map(int, d[2:])
        speaker_data_map[speaker].append((x, y))

print("Generating embedding matrix...", end="")
sys.stdout.flush()
emb_model = Word2Vec.load_word2vec_format(args.embedding_file, binary=True)
emb_matrix = np.zeros([len(emb_model.index2word), emb_model.vector_size])
for i, w in enumerate(emb_model.index2word):
    emb_matrix[i, :] = emb_model[w]
print("done.")

print("Building model...", end="")
sys.stdout.flush()
model = Sequential()
embedding = Embedding(input_dim=len(emb_model.index2word),
                      output_dim=emb_model.vector_size,
                      input_length=MAX_FEATS,
                      weights=[emb_matrix])
embedding.params = []
embedding.updates = []
model.add(embedding)
model.add(GRU(output_dim=HIDDEN_LAYER_SIZE, activation="tanh",
              return_sequences=True))
model.add(Dropout(DROPOUT_PROB))
model.add(GRU(output_dim=HIDDEN_LAYER_SIZE, activation="tanh"))
model.add(Dropout(DROPOUT_PROB))
model.add(Dense(output_dim=1, activation="sigmoid"))
model.compile(optimizer=Adam(lr=LEARNING_RATE, clipvalue=GRAD_CLIP),
              loss="binary_crossentropy",
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
fold = 0
init_weights = model.get_weights()
for train_idx, test_idx in KFold(len(speakers), VALIDATION):
    fold += 1
    print("FOLD {}".format(fold))
    X_train, y_train = generate_XY(train_idx, speakers, speaker_data_map)
    X_test, y_test = generate_XY(test_idx, speakers, speaker_data_map)

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
    accs.append(acc)
    print("Fold {} test accuracy: {}".format(fold, acc))

    fold_save_dir = os.path.join(SAVE_PATH, "fold{}".format(fold))
    if not os.path.exists(fold_save_dir):
        os.makedirs(fold_save_dir)
    model.save_weights(os.path.join(fold_save_dir, "weights.h5"),
                       overwrite=True)
    print("\n".join(map(str, history.history["acc"])),
          file=open(os.path.join(fold_save_dir, "accs.txt"), "w"))
    print("\n".join(map(str, history.history["loss"])),
          file=open(os.path.join(fold_save_dir, "losses.txt"), "w"))

print("\n".join(map(str, accs)),
      file=open(os.path.join(SAVE_PATH, "accs.txt"), "w"))
final_acc = sum(accs) / VALIDATION
print("Final accuracy: {}".format(final_acc))

