import numpy as np
import os
from collections import defaultdict
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--feature-files", type=str, nargs="+", required=True)
arg_parser.add_argument("--sig-files", type=str, nargs="+", required=True)
arg_parser.add_argument("--names-files", type=str, nargs="+", required=True)
arg_parser.add_argument("--output-file", type=str, required=True)
arg_parser.add_argument("--output-names-file", type=str, default=None)
arg_parser.add_argument("--output-sig-file", type=str, required=True)
args = arg_parser.parse_args()

sig = []
X = defaultdict(list)
sig_offset = 0
for feat_file, sig_file, names_file in zip(args.feature_files, args.sig_files, args.names_files):
	names = []
	with open(names_file) as file:
		for line in file:
			names.append(int(line.strip()))
	with open(feat_file) as file:
		for num, line in enumerate(file):
			X[names[num]].extend(line.strip().split(","))
	num_feats = len(line.strip().split(","))
        with open(sig_file) as file:
                for line in file:
                        sig.append(sig_offset + int(line.strip()))
	sig_offset = sig_offset + num_feats
        print sig_offset



with open(args.output_file, "w") as file:
	for num in names:
		file.write(",".join(X[num]))
		file.write("\n")

with open(args.output_sig_file, "w") as file:
        for num in sig:
                file.write(str(num))
                file.write("\n")

if args.output_names_file is not None:
	with open(args.output_names_file, "w") as file:
        	for num in names:
                	file.write(str(num))
	                file.write("\n")

