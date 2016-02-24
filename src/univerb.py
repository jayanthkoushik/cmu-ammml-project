# coding=utf-8
# univerb.py: unimodal verb-based classifier.

from __future__ import print_function
import sys
import argparse
import os
from collections import defaultdict

import numpy as np
from models.gru2 import GRU2
from datetime import datetime
from keras.preprocessing import sequence
from keras.optimizers import Adam


FEATS_FILE = "data/feats/transc.txt"
SAVE_PATH = "data/saves/univerb/"
SPLIT_DIR = "data/perssplit"
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

file_data_map = defaultdict(list)
with open(FEATS_FILE) as ff:
    for line in ff:
        d = line.strip().split(" ")
        file_id = d[0]
        speaker = d[1]
        y = int(d[2])
        x = map(int, d[3:])
        file_data_map[file_id].append((x, y))

Xs = {}
ys = {}
for typ in ["train", "val", "test"]:
    Xs[typ] = []
    ys[typ] = []
    vids_file = os.path.join(SPLIT_DIR, "{}.txt".format(typ))
    with open(vids_file) as vf:
        for line in vf:
            file_id = line.strip()
            for x, y in file_data_map[file_id]:
                Xs[typ].append(x)
                ys[typ].append(y)
    Xs[typ] = sequence.pad_sequences(Xs[typ], maxlen=MAX_FEATS, padding="post",
                                    truncating="post")
    ys[typ] = np.array(ys[typ])

if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    arg_parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    args = arg_parser.parse_args()

    print("Building model...", end="")
    sys.stdout.flush()
    model = GRU2(VOCAB_SIZE, EMBEDDING_SIZE, MAX_FEATS, HIDDEN_LAYER_SIZE,
                 DROPOUT_PROB)
    model.compile(optimizer=Adam(lr=args.lr, clipvalue=GRAD_CLIP),
                  loss="binary_crossentropy",
                  class_mode="binary")
    print("done")

    history = model.fit(Xs["train"], ys["train"], batch_size=args.batch_size,
                        nb_epoch=args.epochs, show_accuracy=True)

    _, acc = model.evaluate(Xs["val"], ys["val"], batch_size=args.batch_size,
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
        "vocab_size": VOCAB_SIZE,
        "embedding_size": EMBEDDING_SIZE,
        "max_feats": MAX_FEATS,
        "hidden_layer_size": HIDDEN_LAYER_SIZE,
        "dropout_prob": DROPOUT_PROB,
        "final_accuracy": acc
    }
    print(summary, file=open(os.path.join(fold_save_dir, "summary.txt"), "w"))

