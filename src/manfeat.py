# coding=utf-8
# manfeat.py: classifier based on manually extracted features.

from __future__ import print_function
import argparse
import os
import cPickle
import itertools
import sys
from pickle import PicklingError
from datetime import datetime

import numpy as np
from scipy.stats import mode
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.shallow import ShallowNet


SPLIT_DIR = "data/perssplit"
SPLITS = ["train", "val", "test"]
PICKLED_LABEL_FILE = "data/labels.pickle"
PERS_FIELD_NAME = "Answer.q7_persuasive"
LR_DECREASE_AT = 0.8
LR_DECREASE_BY = 10.0


def eval_pred(y, pred):
    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred)
    rec = recall_score(y, pred)
    f1 = f1_score(y, pred)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def lr_schedule(total_epochs, base_lr, epoch):
    if epoch >= total_epochs * LR_DECREASE_AT:
        return base_lr / LR_DECREASE_BY
    else:
        return base_lr


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--feats-file", type=str, required=True)
arg_parser.add_argument("--names-file", type=str, required=True)
arg_parser.add_argument("--sig-feats-file", type=str, default=None)
arg_parser.add_argument("--save-path", type=str, required=True)
arg_parser.add_argument("--train", type=str, choices=["true", "false"], required=True)
arg_parser.add_argument("--weights", type=str, default=None)
arg_parser.add_argument("--lr", type=float, nargs="+", required=True)
arg_parser.add_argument("--epochs", type=int, nargs="+", required=True)
arg_parser.add_argument("--dropout", type=float, nargs="+", required=True)
arg_parser.add_argument("--dense-layers", type=int, nargs="+", required=True)
arg_parser.add_argument("--dense-layer-units", type=int, nargs="+", required=True)
arg_parser.add_argument("--batch-size", type=int, nargs="+", required=True)
arg_parser.add_argument("--ensemble-size", type=int, required=True)
arg_parser.add_argument("--test-ensemble-size", type=int, required=True)
arg_parser.add_argument("--continue-dir", type=str, default=None)
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

with open(args.names_file) as man_feats_names_file, open(args.feats_file) as feats_file:
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

if args.sig_feats_file is not None:
    print("Selecting significant features")
    with open(args.sig_feats_file) as sig_feats_file:
        sig_feats = [int(line.strip()) - 1 for line in sig_feats_file]
    for split in SPLITS:
        Xs[split] = Xs[split][:, sig_feats]

if args.train == "true":
    if not args.continue_dir:
        date = str(datetime.now().date())
        base_save_dir = os.path.join(args.save_path, date)
        os.makedirs(base_save_dir)
        os.makedirs(os.path.join(base_save_dir, "checkpoints"))
        final_train_perfs = {}
        final_val_perfs = {}
    else:
        base_save_dir = args.continue_dir
        final_perfs = {}
        for split in ["train", "val"]:
            try:
                with open(os.path.join(base_save_dir, "checkpoints", "final_{}_perfs_0.pickle".format(split)), "rb") as sf:
                    final_perfs[split] = cPickle.load(sf)
            except PicklingError:
                try:
                    with open(os.path.join(base_save_dir, "checkpoints", "final_{}_perf_1.pickle".format(split)), "rb") as sf:
                        final_perfs[split] = cPickle.load(sf)
                except PicklingError:
                    print("Both copies of final_{}_perfs are corrupt. You're fucked dude! I'm out.".format(split))
                    sys.exit(1)
        final_train_perfs = final_perfs["train"]
        final_val_perfs = final_perfs["val"]

    print(
        "\n".join("{}: {}".format(x[0], x[1])
            for x in zip(
                ["lr", "epochs", "dropout", "dense_layers", "dense_layer_units", "batch_size"],
                [args.lr, args.epochs, args.dropout, args.dense_layers, args.dense_layer_units, args.batch_size]
            )
        ),
        file=open(os.path.join(base_save_dir, "cross_validation_params.txt"), "w")
    )

    turn = 0
    for lr, epochs, dropout, dense_layers, dense_layer_units, batch_size in itertools.product(args.lr, args.epochs, args.dropout, args.dense_layers, args.dense_layer_units, args.batch_size):
        params = lr, epochs, dropout, dense_layers, dense_layer_units, batch_size
        print("LR: {}, EPOCHS: {}, DROPOUT: {}, DENSE LAYERS: {}, DENSE_LAYER_UNITS: {}, BATCH_SIZE: {}".format(*params))
        if params in final_train_perfs and params in final_val_perfs:
            print("Skipping: already have results")
            continue
        save_path = os.path.join(base_save_dir, "lr{};epochs{};dropout{};dense_layers{};dense_layer_units{};batch_size{}".format(*params))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        best_val_perf = {"f1": 0}
        for i in range(args.ensemble_size):
            print("Building model")
            model = ShallowNet(Xs["train"].shape[1], dropout, dense_layers, dense_layer_units, args.weights)
            model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy")
            print("Model built")

            history = model.fit(
                X=Xs["train"],
                y=ys["train"],
                batch_size=batch_size,
                nb_epoch=epochs,
                verbose=1,
                validation_data=(Xs["val"], ys["val"]),
                shuffle=True,
                show_accuracy=True,
                callbacks=[LearningRateScheduler(lambda e: lr_schedule(epochs, lr, e))]
            )

            print("\n".join(map(str, history.history["acc"])), file=open(os.path.join(save_path, "train_accs{}.txt".format(i)), "w"))
            print("\n".join(map(str, history.history["loss"])), file=open(os.path.join(save_path, "train_losses{}.txt".format(i)), "w"))
            print("\n".join(map(str, history.history["val_acc"])), file=open(os.path.join(save_path, "val_accs{}.txt".format(i)), "w"))
            print("\n".join(map(str, history.history["val_loss"])), file=open(os.path.join(save_path, "val_losses{}.txt".format(i)), "w"))

            val_pred = model.predict_classes(X=Xs["val"], batch_size=batch_size, verbose=0)
            val_perf = eval_pred(ys["val"], val_pred)
            if val_perf["f1"] >= best_val_perf["f1"]:
                best_val_perf = val_perf
                train_pred = model.predict_classes(X=Xs["train"], batch_size=batch_size, verbose=0)
                best_train_perf = eval_pred(ys["train"], train_pred)

        final_train_perfs[params] = best_train_perf
        final_val_perfs[params] = best_val_perf
        print("final train perf: acc {}, f1 {}; final val perf: acc {}, f1 {}".format(final_train_perfs[params]["acc"], final_train_perfs[params]["f1"], final_val_perfs[params]["acc"], final_val_perfs[params]["f1"]))

        with open(os.path.join(base_save_dir, "checkpoints", "final_train_perfs_{}.pickle".format(turn)), "wb") as sf:
            cPickle.dump(final_train_perfs, sf)
        with open(os.path.join(base_save_dir, "checkpoints", "final_val_perfs_{}.pickle".format(turn)), "wb") as sf:
            cPickle.dump(final_val_perfs, sf)
        turn = 1 - turn

    print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), final_train_perfs.items())), file=open(os.path.join(base_save_dir, "final_train_perfs.txt"), "w"))
    print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), final_val_perfs.items())), file=open(os.path.join(base_save_dir, "final_val_perfs.txt"), "w"))

    best_params = max(final_val_perfs, key=lambda x: final_val_perfs[x]["f1"])
else:
    best_params = args.lr[0], args.epochs[0], args.dropout[0], args.dense_layers[0], args.dense_layer_units[0], args.batch_size[0]

best_lr, best_epochs, best_dropout, best_dense_layers, best_dense_layer_units, best_batch_size = best_params

if args.train == "true":
    save_path = os.path.join(base_save_dir, "best_params")
    os.makedirs(save_path)

if args.train == "true":
    best_val_perf = {"f1": 0}
    for i in range(args.test_ensemble_size):
        print("Building model")
        model = ShallowNet(Xs["train"].shape[1], best_dropout, best_dense_layers, best_dense_layer_units, args.weights)
        model.compile(optimizer=Adam(lr=best_lr), loss="binary_crossentropy")
        print("Model built")

        history = model.fit(
            X=Xs["train"],
            y=ys["train"],
            batch_size=best_batch_size,
            nb_epoch=best_epochs,
            verbose=1,
            validation_data=(Xs["val"], ys["val"]),
            shuffle=True,
            show_accuracy=True,
            callbacks=[LearningRateScheduler(lambda e: lr_schedule(best_epochs, best_lr, e))]
        )

        print("\n".join(map(str, history.history["acc"])), file=open(os.path.join(save_path, "train_accs{}.txt".format(i)), "w"))
        print("\n".join(map(str, history.history["loss"])), file=open(os.path.join(save_path, "train_losses{}.txt".format(i)), "w"))
        print("\n".join(map(str, history.history["val_acc"])), file=open(os.path.join(save_path, "val_accs{}.txt".format(i)), "w"))
        print("\n".join(map(str, history.history["val_loss"])), file=open(os.path.join(save_path, "val_losses{}.txt".format(i)), "w"))

        val_pred = model.predict_classes(X=Xs["val"], batch_size=best_batch_size, verbose=0)
        val_perf = eval_pred(ys["val"], val_pred)
        if val_perf["f1"] >= best_val_perf["f1"]:
            best_val_perf = val_perf
            best_model = model
else:
    print("Building model")
    best_model = ShallowNet(Xs["train"].shape[1], best_dropout, best_dense_layers, best_dense_layer_units, args.weights)
    best_model.compile(optimizer=Adam(lr=best_lr), loss="binary_crossentropy")
    print("Model built")

final_pred = best_model.predict_classes(X=Xs["test"], batch_size=best_batch_size, verbose=0)
test_perf = eval_pred(ys["test"], final_pred)
print("Test labels:")
print(list(ys["test"]))
print("Predictions")
print([x[0] for x in final_pred])
print("Test perf: {}".format(test_perf))

if args.train == "true":
    best_model.save_weights(os.path.join(save_path, "best_weights.h5"), overwrite=True)
    print(
        "\n".join("{}: {}".format(x[0], x[1])
            for x in zip(
                ["lr", "epochs", "dropout", "dense_layers", "dense_layer_units", "batch_size"],
                [args.lr, args.epochs, args.dropout, args.dense_layers, args.dense_layer_units, args.batch_size]
            )
        ),
        file=open(os.path.join(base_save_dir, "cross_validation_params.txt"), "w")
    )
    summary = {
        "best_lr": best_lr,
        "best_epochs": best_epochs,
        "best_dropout":  best_dropout,
        "best_dense_layers": best_dense_layers,
        "best_dense_layer_units": best_dense_layer_units,
        "best_batch_size": best_batch_size,
        "ensemble_size": args.ensemble_size,
        "test_ensemble_size": args.test_ensemble_size,
        "test_perf": test_perf
    }
    print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), summary.items())), file=open(os.path.join(base_save_dir, "summary.txt"), "w"))
