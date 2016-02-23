# coding=utf-8
# extract_transc_feats.py: extract word features from transcriptions and
# persuasiveness from the labels.

from __future__ import print_function
import glob
import re
import itertools
import cPickle
from collections import Counter

import nltk


VOCAB_FILE = "data/vocab.txt"
TRANSC_RAW_DIR = "data/raw/transc"
TRANSC_FEATS_FILE = "data/feats/transc.txt"
BLACKLIST = ["244623.txt", "243646.txt", "181504.txt", "221153.txt"]
PICKLED_LABEL_FILE = "data/labels.pickle"
PICKLED_SPEAKER_FILE = "data/speakers.pickle"
PERS_FIELD_NAME = "Answer.q7_persuasive"
WORD_IGNORE_THRESHOLD = 0


def get_toks(fname):
    with open(fname) as f:
        transc_str = f.read()
    transc_str = transc_str.replace("\r\n", " ")
    transc_str = transc_str.replace("\r\r", " ")
    transc_str = transc_str.replace("\n", " ")
    transc_str = transc_str.replace("-", " ")

    # Convert the transcription to lower case, but make
    # filler markers (like "umm", "uhh" etc.) upper case.
    transc_str = transc_str.lower()
    m = re.findall("[{(]([^\s]*?)[)}]", transc_str)
    for word in m:
        transc_str = re.sub("[({{]{}[)}}]".format(word),
                            " " + word.upper() + " ", transc_str)

    # Construct the feature vector.
    return nltk.word_tokenize(transc_str)


with open(VOCAB_FILE) as vf:
    vocab = {line.strip(): i for (i, line) in enumerate(vf)}
max_feat_vec_len = 0
feat_vecs_written = 0

with open(TRANSC_FEATS_FILE, "w") as feats_file:
    word_counter = Counter()
    for fname in glob.iglob(TRANSC_RAW_DIR + "/*"):
        toks = get_toks(fname)
        for tok in toks:
            if tok in vocab:
                word_counter[tok] += 1
    filtered_vocab = {}
    for tok, c in word_counter.items():
        if c > WORD_IGNORE_THRESHOLD:
            filtered_vocab[tok] = len(filtered_vocab) + 1

    with open(PICKLED_LABEL_FILE, "rb") as lf:
        labels_map = cPickle.load(lf)
    with open(PICKLED_SPEAKER_FILE, "rb") as sf:
        speakers_map = cPickle.load(sf)
    for fname in glob.iglob(TRANSC_RAW_DIR + "/*"):
        if any(fname.endswith(s) for s in BLACKLIST):
            print("Ignoring '{}' (blacklisted)".format(fname))
            continue
        file_id = fname.split("/")[-1].split(".")[0]
        try:
            score = labels_map[file_id][PERS_FIELD_NAME]
        except KeyError:
            print("Label not found for '{}'. Skipping.".format(fname))
            continue
        if score <= 2.5:
            score = 0
        elif score >= 5.5:
            score = 1
        else:
            print("Score not extreme for '{}'. Skipping".format(fname))
            continue

        toks = get_toks(fname)
        feat_vec = []
        for tok in toks:
            if tok in filtered_vocab:
                feat_vec.append(filtered_vocab[tok])
            else:
                feat_vec.append(1 + len(filtered_vocab))

        if len(feat_vec) > max_feat_vec_len:
            max_feat_vec_len = len(feat_vec)

        # Appending score at the beginning of the feature vector.
        feat_vec.insert(0, score)
        # Appending file id and speaker at the beginning of the feature vector.
        feat_vec.insert(0, speakers_map[file_id])
        feat_vec.insert(0, file_id)

        print(" ".join(itertools.imap(str, feat_vec)), file=feats_file)
        feat_vecs_written += 1

print("Filtered vocab size: {}".format(len(filtered_vocab)))
print("{} features written".format(feat_vecs_written))
print("Max feature vector length: {}".format(max_feat_vec_len))

