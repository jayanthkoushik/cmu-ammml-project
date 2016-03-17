# coding=utf-8
# avimg.py: unimodal average image based classifier.

from __future__ import print_function
import argparse
import os
import random
import glob
import cPickle
import shutil
import gc
import time
from datetime import datetime

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.vgg16 import VGG16


SPLIT_DIR = "data/perssplit"
PICKLED_LABEL_FILE = "data/labels.pickle"
PERS_FIELD_NAME = "Answer.q7_persuasive"
DEFAULT_LEARNING_RATES = [0.0001]
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 100


with open(PICKLED_LABEL_FILE, "rb") as lf:
    labels_map = cPickle.load(lf)


def generate_batch(batch_ims):
    """Generate a batch (X, y) from a list of images."""
    batch_X = np.zeros((len(batch_ims), 3, 224, 224))
    batch_y = np.zeros((len(batch_ims), 1))
    for i, im_file in enumerate(batch_ims):
        img = imread(im_file).astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1)).astype(np.float32)
        batch_X[i, :, :, :] = img

        file_id = im_file.split("/")[-1].split(".")[0]
        score = labels_map[file_id][PERS_FIELD_NAME]
        if score >= 5.5:
            batch_y[i] = 1
    return (batch_X, batch_y)


class RandomBatchGenerator(object):

    """Generate random batches of data."""

    def __init__(self, batch_size, typs, imdir, augment, randomize):
        # typs should be a list of "train", "val", or "test".
        self._batch_size = batch_size
        self._randomize = randomize
        self._idx = 0
        if augment is True:
            self._datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0,
                height_shift_range=0,
                shear_range=0,
                horizontal_flip=True,
                vertical_flip=True
            )
        else:
            self._datagen = None
        self._ims = []
        for typ in set(typs):
            vids_file = os.path.join(SPLIT_DIR, "{}.txt".format(typ))
            with open(vids_file) as vf:
                self._ims.extend([os.path.join(imdir, line.strip() + ".jpg") for line in vf])

    def __iter__(self):
        return self

    def next(self):
        if self._randomize:
            batch_ims = random.sample(self._ims, self._batch_size)
        else:
            batch_ims = self._ims[self._idx:self._idx+self._batch_size]
            self._idx += self._batch_size
            if self._idx >= len(self._ims):
                self._idx = 0
        batch_X, batch_y = generate_batch(batch_ims)
        if self._datagen is None:
            return batch_X, batch_y
        else:
            return next(self._datagen.flow(
                X=batch_X,
                y=batch_y,
                batch_size=self._batch_size,
                shuffle=False
            ))


class BatchLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.accs.append(logs.get("acc"))


def eval_model(model, generator):
    ys = []
    preds = []
    done = 0
    while done < len(generator._ims):
        X, y = next(generator)
        pred = model.predict_classes(X=X, batch_size=generator._batch_size, verbose=0)
        ys.extend(y)
        preds.extend(pred)
        done += generator._batch_size
    acc = accuracy_score(ys, preds)
    prec = precision_score(ys, preds)
    rec = recall_score(ys, preds)
    f1 = f1_score(ys, preds)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--imdir", type=str, required=True)
    arg_parser.add_argument("--vgg-weights", type=str, required=True)
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LEARNING_RATES)
    arg_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    arg_parser.add_argument("--train", type=str, choices=["true", "false"], required=True)
    arg_parser.add_argument("--default-arch-weights", type=str, choices=["true", "false"], required=True)
    arg_parser.add_argument("--augment", type=str, choices=["true", "false"], required=True)
    args = arg_parser.parse_args()

    default_arch_weights = args.default_arch_weights == "true"

    if args.train == "true":
        date = str(datetime.now().date())
        base_save_dir = os.path.join(args.save_path, date)
        os.makedirs(base_save_dir)

        shutil.copyfile(os.path.join(os.path.dirname(os.path.realpath(__file__)), "models", "vgg16.py"), os.path.join(base_save_dir, "vgg16.py"))

        final_train_perfs = {}
        final_val_perfs = {}
        for lr in args.lrs:
            print("LR: {}".format(lr))
            print("Building model")
            model = VGG16(args.vgg_weights, default_arch_weights)
            model.compile(optimizer=Adam(lr=lr), loss="binary_crossentropy")
            print("Model built")

            save_path = os.path.join(base_save_dir, "lr{}".format(lr))
            os.makedirs(save_path)

            train_generator = RandomBatchGenerator(args.batch_size, ["train"], args.imdir, args.augment=="true", True)
            val_generator = RandomBatchGenerator(args.batch_size, ["val"], args.imdir, args.augment=="true", True)

            batch_hist_clbk = BatchLossHistory()

            history = model.fit_generator(
                generator=train_generator,
                samples_per_epoch=len(train_generator._ims),
                nb_epoch=args.epochs,
                verbose=1,
                show_accuracy=True,
                callbacks=[batch_hist_clbk],
                validation_data=val_generator,
                nb_val_samples=len(val_generator._ims),
                nb_worker=2
            )

            fixed_train_generator = RandomBatchGenerator(args.batch_size, ["train"], args.imdir, False, False)
            fixed_val_generator = RandomBatchGenerator(args.batch_size, ["val"], args.imdir, False, False)

            final_train_perfs[lr] = eval_model(model, fixed_train_generator)
            final_val_perfs[lr] = eval_model(model, fixed_val_generator)
            print("LR {} final train perf: acc {}, f1 {}; final val perf: acc {}, f1 {}".format(lr, final_train_perfs[lr]["acc"], final_train_perfs[lr]["f1"], final_val_perfs[lr]["acc"], final_val_perfs[lr]["f1"]))

            model.save_weights(os.path.join(save_path, "weights.h5"), overwrite=True)
            print("\n".join(map(str, history.history["acc"])), file=open(os.path.join(save_path, "epoch_train_accs.txt"), "w"))
            print("\n".join(map(str, history.history["loss"])), file=open(os.path.join(save_path, "epoch_train_losses.txt"), "w"))
            print("\n".join(map(str, history.history["val_acc"])), file=open(os.path.join(save_path, "epoch_val_accs.txt"), "w"))
            print("\n".join(map(str, history.history["val_loss"])), file=open(os.path.join(save_path, "epoch_val_losses.txt"), "w"))
            print("\n".join(map(str, batch_hist_clbk.accs)), file=open(os.path.join(save_path, "batch_accs.txt"), "w"))
            print("\n".join(map(str, batch_hist_clbk.losses)), file=open(os.path.join(save_path, "batch_losses.txt"), "w"))

            print("Freeing memory")
            del model
            del train_generator
            del val_generator
            del batch_hist_clbk
            del history
            del fixed_train_generator
            del fixed_val_generator
            gc.collect()
            time.sleep(60)

        print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), final_train_perfs.items())), file=open(os.path.join(base_save_dir, "final_train_perfs.txt"), "w"))
        print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), final_val_perfs.items())), file=open(os.path.join(base_save_dir, "final_val_perfs.txt"), "w"))
        
        best_lr = max(final_val_perfs, key=lambda x: final_val_perfs[x]["f1"])
        print("Best learning rate: {}".format(best_lr))
    else:
        best_lr = DEFAULT_LEARNING_RATES[0]
    
    print("Building model")
    model = VGG16(args.vgg_weights, default_arch_weights)
    model.compile(optimizer=Adam(lr=best_lr), loss="binary_crossentropy")
    print("Model built")

    if args.train == "true":
        print("Training best model on training and validation set")
        save_path = os.path.join(base_save_dir, "best_lr")
        os.makedirs(save_path)

        batch_hist_clbk = BatchLossHistory()

        train_val_generator = RandomBatchGenerator(args.batch_size, ["train", "val"],args.imdir, args.augment=="true", True)

        history = model.fit_generator(
            generator=train_val_generator,
            samples_per_epoch=len(train_val_generator._ims),
            nb_epoch=args.epochs,
            verbose=1,
            show_accuracy=True,
            callbacks=[batch_hist_clbk],
            nb_worker=2
        )

        model.save_weights(os.path.join(save_path, "weights.h5"), overwrite=True)
        print("\n".join(map(str, history.history["acc"])), file=open(os.path.join(save_path, "epoch_train_accs.txt"), "w"))
        print("\n".join(map(str, history.history["loss"])), file=open(os.path.join(save_path, "epoch_train_losses.txt"), "w"))
        print("\n".join(map(str, batch_hist_clbk.accs)), file=open(os.path.join(save_path, "batch_accs.txt"), "w"))
        print("\n".join(map(str, batch_hist_clbk.losses)), file=open(os.path.join(save_path, "batch_losses.txt"), "w"))

        os.remove(os.path.join(save_path, "checkpoint.h5"))

    test_generator = RandomBatchGenerator(args.batch_size, ["test"], args.imdir, False, False)
    test_perf = eval_model(model, test_generator)
    print("Test perf: {}".format(test_perf))

    if args.train == "true":
        summary = {
            "best_lr": best_lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_perf": test_perf,
        }
        print("\n".join(map(lambda x: "{}: {}".format(x[0], x[1]), summary.items())), file=open(os.path.join(base_save_dir, "summary.txt"), "w"))

