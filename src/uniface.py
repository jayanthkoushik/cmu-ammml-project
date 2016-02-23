# coding=utf-8
# uniface.py: unimodal face image based classifier.

from __future__ import print_function
import argparse
import sys
import os
import random
import glob
import shelve

import numpy as np
from vgg16 import VGG16
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from scipy.misc import imread


SAVE_PATH = "data/saves/uniface"
SPLIT_DIR = "data/perssplit"
SHELVED_LABEL_FILE = "data/labels.db"
PERS_FIELD_NAME = "Answer.q7_persuasive"
LEARNING_RATE = 0.0001
GRAD_CLIP = 3
DEFAULT_EPOCHS = 1
DEFAULT_BATCH_SIZE = 100

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--imdir", type=str, required=True)
arg_parser.add_argument("--vgg-weights", type=str, required=True)
arg_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
args = arg_parser.parse_args()

print("Building model...", end="")
sys.stdout.flush()
model = VGG16(args.vgg_weights)
model.compile(optimizer=Adam(lr=LEARNING_RATE, clipvalue=GRAD_CLIP),
              loss="binary_crossentropy",
              class_mode="binary")
print("done")

labels_map = shelve.open(SHELVED_LABEL_FILE)


class BatchGenerator(object):

    """Generate batches of training data."""

    def __init__(self, batch_size, typ, imdir, sequential):
        # typ should be "train", "val", or "test".
        self._batch_size = batch_size
        self._ims = []
        self._sequential = sequential
        self._idx = 0
        vids_file = os.path.join(SPLIT_DIR, "{}.txt".format(typ))
        with open(vids_file) as vf:
            for line in vf:
                vid_ims = os.path.join(imdir, line.strip(), "*")
                self._ims.extend(glob.glob(vid_ims))

    def __iter__(self):
        return self

    def next(self):
        global labels_map
        if self._sequential:
            # Iterate over the images in sequence.
            batch_ims = self._ims[self._idx:self._idx+self._batch_size]
            self._idx = self._idx + self._batch_size
            if self._idx >= self._batch_size:
                self._idx = 0
        else:
            # Draw a random sample from the images.
            batch_ims = random.sample(self._ims, self._batch_size)
        batch_X = np.zeros((self._batch_size, 3, 224, 224))
        batch_y = np.zeros((self._batch_size, 1))
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


train_generator = BatchGenerator(args.batch_size, "train", args.imdir, False)
val_generator = BatchGenerator(args.batch_size, "val", args.imdir, False)
ckpt_clbk = ModelCheckpoint(
    filepath=os.path.join(SAVE_PATH, "checkpoint.h5"),
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
    nb_val_samples=len(val_generator._ims) // 4, # Use a quarter of the data
    nb_worker=1
)
eval_generator = BatchGenerator(args.batch_size, "val", args.imdir, True)
_, acc = model.evaluate_generator(
    generator=eval_generator,
    val_samples=len(eval_generator._ims),
    show_accuracy=True,
    verbose=1
)
print("Final accuracy: {}".format(acc))

print("Saving...", end="")
sys.stdout.flush()
model.save_weights(os.path.join(SAVE_PATH, "weights.h5"), overwrite=True)
print("\n".join(map(str, history.history["acc"])),
      file=open(os.path.join(SAVE_PATH, "accs.txt"), "w"))
print("\n".join(map(str, history.history["loss"])),
      file=open(os.path.join(SAVE_PATH, "losses.txt"), "w"))
print(acc, file=open(os.path.join(SAVE_PATH, "finalacc.txt"), "w"))
print("done.")

