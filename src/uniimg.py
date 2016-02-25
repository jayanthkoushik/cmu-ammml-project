# coding=utf-8
# uniimg.py: unimodal image based classifier.

from __future__ import print_function
import argparse
import sys
import os
import random
import glob
import cPickle
import math
from datetime import datetime

import numpy as np
from models.vgg16 import VGG16
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from scipy.misc import imread


SPLIT_DIR = "data/perssplit"
PICKLED_LABEL_FILE = "data/labels.pickle"
PERS_FIELD_NAME = "Answer.q7_persuasive"
GRAD_CLIP = 3
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 100

with open(PICKLED_LABEL_FILE, "rb") as lf:
    labels_map = cPickle.load(lf)


def generate_batch(batch_ims):
    """Generate a batch (X, y) from a list of images."""
    batch_X = np.zeros((len(batch_ims), 3, 224, 224))
    batch_y = np.zeros((len(batch_ims), 1))
    for i, im_file in enumerate(batch_ims):
        img = imread(im_file).astype("float32")
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img.transpose((2, 0, 1))
        batch_X[i, :, :, :] = img

        file_id = im_file.split("/")[-1].split("_")[0]
        score = labels_map[file_id][PERS_FIELD_NAME]
        if score >= 5.5:
            batch_y[i] = 1
    return (batch_X, batch_y)


class RandomBatchGenerator(object):

    """Generate random batches of data."""

    def __init__(self, batch_size, typ, imdir, augment):
        # typ should be "train", "val", or "test".
        self._batch_size = batch_size
        self._ims = []
        self._idx = 0
        if augment is True:
            self._datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.25,
                height_shift_range=0.25,
                shear_range=0.1,
                horizontal_flip=True,
                vertical_flip=True
            )
        else:
            self._datagen = None
        vids_file = os.path.join(SPLIT_DIR, "{}.txt".format(typ))
        with open(vids_file) as vf:
            for line in vf:
                vid_ims = os.path.join(imdir, line.strip(), "*")
                self._ims.extend(glob.glob(vid_ims))

    def __iter__(self):
        return self

    def next(self):
        batch_ims = random.sample(self._ims, self._batch_size)
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


class VidBatchGenerator(object):

    """Generate batches of data corresponding to a video."""

    def __init__(self, batch_size, vid, imdir):
        self._batch_size = batch_size
        self._idx = 0
        vid_ims = os.path.join(imdir, vid, "*")
        self._ims = glob.glob(vid_ims)

    def __iter__(self):
        return self

    def next(self):
        if self._idx >= len(self._ims):
            self._idx = 0
        batch_ims = self._ims[self._idx:self._idx+self._batch_size]
        self._idx = self._idx + self._batch_size
        return generate_batch(batch_ims)


def eval_model_vid(model, batch_size, vid, imdir):
    """Evaluate a model on a single video."""
    vid_batch_generator = VidBatchGenerator(batch_size, vid, imdir)
    num_ims = len(vid_batch_generator._ims)
    l, acc = model.evaluate_generator(
        generator=vid_batch_generator,
        val_samples=num_ims,
        show_accuracy=True,
        verbose=1
    )
    return l, acc, num_ims


def eval_model(model, batch_size, typ, imdir):
    """Evaluate a model. "typ" should be "train", "val", or "test"."""
    vids_file = os.path.join(SPLIT_DIR, "{}.txt".format(typ))
    total_vids = 0
    correct_vids = 0
    total_ims = 0
    correct_ims = 0
    with open(vids_file) as vf:
        for line in vf:
            _, acc, num_ims = eval_model_vid(model, batch_size, line.strip(), imdir)
            total_vids += 1
            if acc >= 0.5:
                correct_vids += 1
            total_ims += num_ims
            correct_ims += math.floor(acc * num_ims)
    vid_acc = float(correct_vids) / total_vids
    im_acc = float(correct_ims) / total_ims
    return vid_acc, im_acc


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--imdir", type=str, required=True)
    arg_parser.add_argument("--vgg-weights", type=str, required=True)
    arg_parser.add_argument("--save-path", type=str, required=True)
    arg_parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    arg_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    arg_parser.add_argument("--train", type=str, choices=["true", "false"],
                            required=True)
    arg_parser.add_argument("--default-arch-weights", type=str,
                            choices=["true", "false"], required=True)
    arg_parser.add_argument("--augment", type=str,
                            choices=["true", "false"], required=True)
    args = arg_parser.parse_args()

    print("Building model...", end="")
    sys.stdout.flush()
    default_arch_weights = args.default_arch_weights == "true"
    model = VGG16(args.vgg_weights, default_arch_weights)
    model.compile(optimizer=Adam(lr=args.lr, clipvalue=GRAD_CLIP),
                  loss="binary_crossentropy",
                  class_mode="binary")
    print("done")

    if args.train == "true":
        date = str(datetime.now().date())
        args.save_path = os.path.join(args.save_path, date)
        os.makedirs(args.save_path)

        train_generator = RandomBatchGenerator(args.batch_size, "train",
                                               args.imdir, args.augment=="true")
        val_generator = RandomBatchGenerator(args.batch_size, "val",
                                             args.imdir, args.augment=="true")
        ckpt_clbk = ModelCheckpoint(
            filepath=os.path.join(args.save_path, "checkpoint.h5"),
            verbose=1,
            save_best_only=False
        )
        history = model.fit_generator(
            generator=train_generator,
            samples_per_epoch=len(train_generator._ims),
            nb_epoch=args.epochs,
            verbose=1,
            show_accuracy=True,
            callbacks=[ckpt_clbk],
            validation_data=val_generator,
            nb_val_samples=len(val_generator._ims) // 4,
            nb_worker=1
        )

    train_vid_acc, train_im_acc = eval_model(model, args.batch_size, "train",
                                            args.imdir)
    val_vid_acc, val_im_acc = eval_model(model, args.batch_size, "val", args.imdir)
    print("Training: video acc.: {}, image acc.: {}".format(train_vid_acc,
                                                            train_im_acc))
    print("Validation: video acc.: {}, image acc.: {}".format(val_vid_acc,
                                                            val_im_acc))

    if args.train == "true":
        print("Saving...", end="")
        sys.stdout.flush()
        model.save_weights(os.path.join(args.save_path, "weights.h5"),
                           overwrite=True)
        print("\n".join(map(str, history.history["acc"])),
              file=open(os.path.join(args.save_path, "accs.txt"), "w"))
        print("\n".join(map(str, history.history["loss"])),
              file=open(os.path.join(args.save_path, "losses.txt"), "w"))
        summary = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_vid_acc": train_vid_acc,
            "train_im_acc": train_im_acc,
            "val_vid_acc": val_vid_acc,
            "val_im_acc": val_im_acc
        }
        print(summary, file=open(os.path.join(args.save_path, "summary.txt"), "w"))
        print("done.")

