# coding=utf-8
# manfeat.py: classifier based on manually extracted features.

from __future__ import print_function
import argparse
import os
import cPickle
import gc
import time
import shutil
from datetime import datetime

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.shallow import ShallowNet


SPLIT_DIR = "data/perssplit"
SPLITS = ["train", "val", "test"]
PICKLED_LABEL_FILE = "data/labels.pickle"
PERS_FIELD_NAME = "Answer.q7_persuasive"
MAN_FEATS_NAMES_FILENAME = "data/man_feats/names.txt"


class BatchLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.accs.append(logs.get("acc"))


def eval_model(model, batch_size, X, y):
    pred = model.predict_classes(X=x, batch_size=batch_size, verbose=0)
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    f1 = f1_score(y, pred)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--feats-file", type=str, required=True)
arg_parser.add_argument("--save-path", type=str, required=True)
arg_parser.add_argument("--lrs", type=float, nargs="+", required=True)
arg_parser.add_argument("--epochs", type=int, required=True)
arg_parser.add_argument("--batch-size", type=int, required=True)
arg_parser.add_argument("--train", type=str, choices=["true", "false"], required=True)
arg_parser.add_argument("--weights", type=str, default=None)
args = arg_parser.parse_args()

with open(PICKLED_LABEL_FILE, "rb") as lf:
    labels_map = cPickle.load(lf)

name_splits = {}
Xs = {}
ys = {}
for split in SPLITS:
    with open(os.path.join(SPLIT_DIR, "{}.txt".format(split))) as split_file:
        for line in split_file:
            name_splits[line.strip()] = split
    Xs[split] = []
    ys[split] = []

with open(MAN_FEATS_NAMES_FILENAME) as man_feats_names_file, open(args.feats_file) as feats_file:
    for name_line, feat_line in zip(man_feats_names_file, feats_file):
        name = name_line.strip()
        feats = map(float, feat_line.strip().split(","))
        split = name_splits[name]
        Xs[split].append(feats)
        score = labels_map[name][PERS_FIELD_NAME]
        if score >= 5.5:
            ys[split].append(1)
        else:
            ys[split].append(0)

for split in SPLITS:
    Xs[split] = np.array(Xs[split])
    ys[split] = np.array(ys[split])

xmean = np.mean(Xs["train"], axis=0)
for split in SPLITS:
    Xs[split] -= xmean

if args.train == "true":
    date = str(datetime.now().date())
    base_save_dir = os.path.join(args.save_path, date)
    os.makedirs(base_save_dir)

    final_train_perfs = {}
    final_val_perfs = {}
    for lr in args.lrs:
        print("LR: {}".format(lr))
        print("Building model")
        model = ShallowNet(args.weights, Xs["train"].shape[1])
        model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy")
        print("Model built")

        save_path = os.path.join(base_save_dir, "lr{}".format(lr))
        os.makedirs(save_path)

        batch_hist_clbk = BatchLossHistory()

        history = model.fit(
            X=Xs["train"],
            y=ys["train"],
            batch_size=args.batch_size,
            nb_epoch=args.epochs,
            verbose=1,
            callbacks=[batch_hist_clbk],
            validation_data=(Xs["val"], ys["val"]),
            shuffle=True,
            show_accuracy=True,
        )

        final_train_perfs[lr] = eval_model(model, args.batch_size, Xs["train"], ys["train"])
        final_val_perfs[lr] = eval_model(model, args.batch_size, Xs["val"], ys["val"])
        print("LR {} final train perf: acc {}, f1 {}; final val perf: acc {}, f1 {}".format(lr, final_train_perfs[lr]["acc"], final_train_perfs[lr]["f1"], final_val_perfs[lr]["acc"], final_val_perfs[lr]["f1"]))

        print("\n".join(map(str, history.history["acc"])), file=open(os.path.join(save_path, "epoch_train_accs.txt"), "w"))
        print("\n".join(map(str, history.history["loss"])), file=open(os.path.join(save_path, "epoch_train_losses.txt"), "w"))
        print("\n".join(map(str, history.history["val_acc"])), file=open(os.path.join(save_path, "epoch_val_accs.txt"), "w"))
        print("\n".join(map(str, history.history["val_loss"])), file=open(os.path.join(save_path, "epoch_val_losses.txt"), "w"))
        print("\n".join(map(str, batch_hist_clbk.accs)), file=open(os.path.join(save_path, "batch_accs.txt"), "w"))
        print("\n".join(map(str, batch_hist_clbk.losses)), file=open(os.path.join(save_path, "batch_losses.txt"), "w"))

        print("Freeing memory")
        del model
        del batch_hist_clbk
        del history
        gc.collect()
        time.sleep(60)

    print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), final_train_perfs.items())), file=open(os.path.join(base_save_dir, "final_train_perfs.txt"), "w"))
    print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), final_val_perfs.items())), file=open(os.path.join(base_save_dir, "final_val_perfs.txt"), "w"))

    best_lr = max(final_val_perfs, key=lambda x: final_val_perfs[x]["f1"])
    print("Best learning rate: {}".format(best_lr))
else:
    best_lr = 0.0001

print("Building model")
model = ShallowNet(args.weights, Xs["train"].shape[1])
model.compile(optimizer=Adam(lr=best_lr), loss="binary_crossentropy")
print("Model built")

if args.train == "true":
    print("Training best model on training and validation set")
    save_path = os.path.join(base_save_dir, "best_lr")
    os.makedirs(save_path)

    shutil.copy(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "shallow.py"), base_save_dir)

    batch_hist_clbk = BatchLossHistory()

    history = model.fit(
        X=np.concatenate((Xs["train"], Xs["val"]))
        y=np.concatenate((ys["train"], ys["val"]))
        batch_size=args.batch_size,
        nb_epoch=args.epochs,
        verbose=1,
        callbacks=[batch_hist_clbk],
        shuffle=True,
        show_accuracy=True,
    )

    print("\n".join(map(str, history.history["acc"])), file=open(os.path.join(save_path, "epoch_train_accs.txt"), "w"))
    print("\n".join(map(str, history.history["loss"])), file=open(os.path.join(save_path, "epoch_train_losses.txt"), "w"))
    print("\n".join(map(str, batch_hist_clbk.accs)), file=open(os.path.join(save_path, "batch_accs.txt"), "w"))
    print("\n".join(map(str, batch_hist_clbk.losses)), file=open(os.path.join(save_path, "batch_losses.txt"), "w"))

test_perf = eval_model(model, args.batch_size, Xs["test"], ys["test"])
print("Test perf: {}".format(test_perf))

if args.train == "true":
    summary = {
        "best_lr": best_lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "test_perf": test_perf,
    }
    print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), summary.items())), file=open(os.path.join(base_save_dir, "summary.txt"), "w"))
