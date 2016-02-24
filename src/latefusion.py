# coding=utf-8
# latefusion.py: combine the predictions of univerb, uniface, and unikey.

from __future__ import print_function
import argparse
import json
import os
import sys
from datetime import datetime

import univerb
import uniimg
from models.gru2 import GRU2
from models.vgg16 import VGG16
from keras.optimizers import Adam


SPLIT_DIR = "data/perssplit"
DEFAULT_BATCH_SIZE = 100

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--univerb-summary", type=str, required=True)
arg_parser.add_argument("--uniface-weights", type=str, required=True)
arg_parser.add_argument("--unikey-weights", type=str, required=True)
arg_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
arg_parser.add_argument("--face-imdir", type=str, required=True)
arg_parser.add_argument("--key-imdir", type=str, required=True)
arg_parser.add_argument("--evaluate-on", type=str, nargs="+",
                        choices=["train", "test", "val"])
arg_parser.add_argument("--save-path", type=str, required=True)
args = arg_parser.parse_args()
args.evaluate_on = set(args.evaluate_on)

date = str(datetime.now().date())
args.save_path = os.path.join(args.save_path, date)
os.makedirs(args.save_path)

with open(args.univerb_summary) as uvf:
    uvs = uvf.read().replace("'", '"')
    univerb_data = json.loads(uvs)
print("Loading univerb model...", end="")
sys.stdout.flush()
univerb_model = GRU2(
    univerb_data["vocab_size"],
    univerb_data["embedding_size"],
    univerb_data["max_feats"],
    univerb_data["hidden_layer_size"],
    univerb_data["dropout_prob"]
)
univerb_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    class_mode="binary"
)
print("done")

print("Loading uniface model...", end="")
sys.stdout.flush()
uniface_model = VGG16(args.uniface_weights, False)
uniface_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    class_mode="binary"
)
print("done")

print("Loading unikey model...", end="")
sys.stdout.flush()
unikey_model = VGG16(args.unikey_weights, False)
unikey_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    class_mode="binary"
)
print("done")

accs = {}
for typ in args.evaluate_on:
    univerb_preds = univerb_model.predict_classes(
        X=univerb.Xs[typ],
        batch_size=args.batch_size,
        verbose=1
    )

    vids_file = os.path.join(SPLIT_DIR, "{}.txt".format(typ))
    total_vids = 0
    correct_vids = 0
    with open(vids_file) as vf:
        for i, line in enumerate(vf):
            vid = line.strip()
            vid_corrs = 0

            if univerb_preds[i] == univerb.ys[typ][i]:
                vid_corrs += 1

            _, uniface_acc, _ = uniimg.eval_model_vid(
                uniface_model,
                args.batch_size,
                vid,
                args.face_imdir
            )
            if uniface_acc >= 0.5:
                vid_corrs += 1

            _, unikey_acc, _ = uniimg.eval_model_vid(
                unikey_model,
                args.batch_size,
                vid,
                args.key_imdir
            )
            if unikey_acc >= 0.5:
                vid_corrs += 1

            if vid_corrs >= 2:
                correct_vids += 1
            total_vids += 1

    accs[typ] = (100.0 * correct_vids) / total_vids
    print("{}: {}/{} ({}%)".format(typ, correct_vids, total_vids, accs[typ]))

print(accs, file=open(os.path.join(args.save_path, "accs.txt"), "w"))

