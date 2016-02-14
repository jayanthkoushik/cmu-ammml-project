# coding=utf-8
# extract_transc_feats.py: extract word features from transcriptions.

from __future__ import print_function
import glob
import re
import itertools
import shelve
import nltk


VOCAB_FILE = "data/vocab.txt"
TRANSC_RAW_DIR = "data/raw/transc"
TRANSC_FEATS_FILE = "data/feats/transc.txt"
BLACKLIST = ["244623.txt", "243646.txt", "181504.txt", "221153.txt"]
SHELVED_LABEL_FILE = "data/labels.db"


with open(VOCAB_FILE) as vf:
    vocab = {line.strip(): i for (i, line) in enumerate(vf)}
max_feat_vec_len = 0
feat_vecs_written = 0

with open(TRANSC_FEATS_FILE, "w") as feats_file:
    labels_map = shelve.open(SHELVED_LABEL_FILE)
    for fname in glob.iglob(TRANSC_RAW_DIR + "/*"):
        if any(fname.endswith(s) for s in BLACKLIST):
            print("Ignoring '{}' (blacklisted)".format(fname))
            continue
        fileId = fname.split("/")[-1].split(".")[0]
        if fileId not in labels_map:
            print("Label not found for '{}'. Skipping.".format(fname))
            continue
        score = labels_map[fileId]
        if score <= 2.5:
            score = int(0)
        elif score >= 5: 
            score = int(1)
        else:
            print("Score not extreme for '{}'. Skipping".format(fname))
            continue        

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
        toks = nltk.word_tokenize(transc_str)
        feat_vec = []
        for tok in toks:
            if tok in vocab:
                feat_vec.append(vocab[tok])
            else:
                feat_vec.append(len(vocab))

        if len(feat_vec) > max_feat_vec_len:
            max_feat_vec_len = len(feat_vec)

        #Appending score at the beginning of the feature vector
        feat_vec.insert(0, score)

        print(" ".join(itertools.imap(str, feat_vec)), file=feats_file)
        feat_vecs_written += 1
    labels_map.close()

print("{} features written".format(feat_vecs_written))
print("Max feature vector length: {}".format(max_feat_vec_len))

