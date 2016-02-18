# coding=utf-8
# extract_transc_feats.py: extract word features from transcriptions and
# persuasiveness from the labels.

from __future__ import print_function
import glob
import itertools
import shelve
import argparse
import re
import sys

from gensim.models import Word2Vec


TRANSC_RAW_DIR = "data/raw/transc"
TRANSC_FEATS_FILE = "data/feats/transc.txt"
BLACKLIST = ["244623.txt", "243646.txt", "181504.txt", "221153.txt"]
SHELVED_LABEL_FILE = "data/labels.db"
SHELVED_SPEAKER_FILE = "data/speakers.db"
PERS_FIELD_NAME = "Answer.q7_persuasive"

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--embedding-file", type=str, required=True)
args = arg_parser.parse_args()

print("Building word to index map...", end="")
sys.stdout.flush()
emb_model = Word2Vec.load_word2vec_format(args.embedding_file, binary=True)
word2index = {}
word2index = {w: i+1 for i, w in enumerate(emb_model.index2word)}
print("done.")


def get_toks(fname):
    with open(fname) as f:
        transc_str = f.read()
    transc_str = re.sub("[^a-zA-Z']", " ", transc_str)
    return transc_str.split()


max_feat_vec_len = 0
feat_vecs_written = 0

with open(TRANSC_FEATS_FILE, "w") as feats_file:
    labels_map = shelve.open(SHELVED_LABEL_FILE)
    speakers_map = shelve.open(SHELVED_SPEAKER_FILE)
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
            if tok in word2index:
                feat_vec.append(word2index[tok])

        if len(feat_vec) > max_feat_vec_len:
            max_feat_vec_len = len(feat_vec)

        # Appending score at the beginning of the feature vector.
        feat_vec.insert(0, score)
        # Appending speaker at the beginning of the feature vector.
        feat_vec.insert(0, speakers_map[file_id])

        print(" ".join(itertools.imap(str, feat_vec)), file=feats_file)
        feat_vecs_written += 1
    labels_map.close()

print("{} features written".format(feat_vecs_written))
print("Max feature vector length: {}".format(max_feat_vec_len))

