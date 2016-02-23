# coding=utf-8
# extract_pers_split.py: generate train, validation, and test splits
# for persuasion.

from __future__ import print_function
import cPickle
import os
from collections import defaultdict
from random import shuffle


PICKLED_SPEAKER_FILE = "data/speakers.pickle"
TRANSC_FEATS_FILE = "data/feats/transc.txt"
PERS_VIDS_FILE = "data/pers_vids.txt"
SPLIT_DIR = "data/perssplit"
TRAIN_COUNT = 125
VAL_COUNT = 25

if not os.path.exists(SPLIT_DIR):
    os.makedirs(SPLIT_DIR)

# Generate a map from speakers to their videos.
with open(PICKLED_SPEAKER_FILE, "rb") as sf:
    speakers_dict = cPickle.load(sf)
speaker_videos_map = defaultdict(list)
for vid in speakers_dict:
    speaker = speakers_dict[vid]
    speaker_videos_map[speaker].append(vid)

# Generate a list of valid speakers.
speakers = set()
with open(TRANSC_FEATS_FILE) as tf:
    for line in tf:
        speaker = line.strip().split(" ")[0]
        speakers.add(speaker)
speakers = list(speakers)

# Generate a set of valid videos.
videos = set()
with open(PERS_VIDS_FILE) as pf:
    for line in pf:
        videos.add(line.strip())

shuffle(speakers)
trainf = open(os.path.join(SPLIT_DIR, "train.txt"), "w")
valf = open(os.path.join(SPLIT_DIR, "val.txt"), "w")
testf = open(os.path.join(SPLIT_DIR, "test.txt"), "w")

for i, speaker in enumerate(speakers):
    vids = "\n".join(vid for vid in speaker_videos_map[speaker]
                     if vid in videos)
    if i < TRAIN_COUNT:
        print(vids, file=trainf)
    elif i < TRAIN_COUNT + VAL_COUNT:
        print(vids, file=valf)
    else:
        print(vids, file=testf)

