# coding=utf-8
# extract_speakers.py: generate a pickled dictionary mapping
# videos to their speakers.

import cPickle


DATA_FILE = "data/raw/speakers.txt"
OUTPUT_FILE = "data/speakers.pickle"

speakers = {}
with open(DATA_FILE) as df:
    for line in df:
        vid, speaker = line.strip().split("\t")
        speakers[vid] = speaker

with open(OUTPUT_FILE, "wb") as of:
    cPickle.dump(speakers, of, protocol=2)

