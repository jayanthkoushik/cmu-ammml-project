# coding=utf-8
# extract_labels.py: extract the various traits from the raw data file
# and write them to a shelved dictionary.

import csv
import shelve
from collections import defaultdict


DATA_FILE = "data/raw/movie_review_main_HIT_all_icmi.csv"
LABELS_FILE = "data/labels"
KEY_NAME = "Input.videoLink"
TRAITS = {
    # The traits to be extracted.
    "Answer.q10_confident",
    "Answer.q11_passionate",
    "Answer.q12_physAttractive",
    "Answer.q13_starRating",
    "Answer.q15_voicePleasant",
    "Answer.q16_professionalLooking",
    "Answer.q17_dominant",
    "Answer.q18_credible",
    "Answer.q19_vivid",
    "Answer.q20_expertise",
    "Answer.q21_entertaining",
    "Answer.q22_reserved",
    "Answer.q23_trusting",
    "Answer.q24_lazy",
    "Answer.q25_relaxed",
    "Answer.q26_fewArtistic",
    "Answer.q27_outgoing",
    "Answer.q28_findFaults",
    "Answer.q29_thorough",
    "Answer.q30_nervous",
    "Answer.q31_activeImag",
    "Answer.q5_sentiment",
    "Answer.q7_persuasive",
    "Answer.q9_humerous"
}

labels_map = shelve.open(LABELS_FILE)
labels_map.clear()
with open(DATA_FILE, "rb") as dataf:
    reader = csv.DictReader(dataf, delimiter=",")
    for row in reader:
        key = row[KEY_NAME].split("/")[-1].split(".")[0]
        for trait in TRAITS:
            trait_val = float(row[trait].split("_")[0])
            try:
                trait_map = labels_map[key]
            except KeyError:
                trait_map = defaultdict(int)
            trait_map[trait] += trait_val / 3.0
            labels_map[key] = trait_map
labels_map.close()

