import numpy as np
import os
from collections import defaultdict

FEATS_FOLDER = "data/man_feats/mattext"
OUTPUT_FOLDER = "data/early_fusion"
MODALITIES = ["audio", "text", "visual"]
SPLITS = ["train", "test", "val"]

for split in SPLITS:
	X = defaultdict(list)
	filename = "X_all_" + split + ".txt"
	for modality in MODALITIES:
		with open(os.path.join(FEATS_FOLDER, modality, filename), "r") as file:
			for num, line in enumerate(file):
				X[num] = np.append(X[num], line.strip().split(","))
	if os.path.exists(OUTPUT_FOLDER) is False:
		os.makedirs(OUTPUT_FOLDER)
	with open(os.path.join(OUTPUT_FOLDER, filename), "w") as outfile:
		for num in range(len(X)):
			outfile.write(",".join(X[num]))
			outfile.write("\n")

