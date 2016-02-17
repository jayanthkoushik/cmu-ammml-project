# coding=utf-8
# extract_speakers.py: generate a shelved db mapping
# videos to their speakers.

import shelve


DATA_FILE = "data/raw/speakers.txt"
OUTPUT_FILE = "data/speakers"

speaker_db = shelve.open(OUTPUT_FILE)
with open(DATA_FILE) as df:
    for line in df:
        vid, speaker = line.strip().split("\t")
        speaker_db[vid] = speaker
speaker_db.close()

