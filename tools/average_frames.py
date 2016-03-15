# coding=utf-8
# average_frames: average the frames for each video.

from __future__ import print_function
import os
import glob
import argparse
import multiprocessing as mp

import numpy as np
import stsci.image
from scipy.misc import imread, imsave


CHUNKSIZE = 1


def process_video(vid_path):
    vid_name = vid_path.split("/")[-1]
    print(vid_name)
    frame_paths = glob.iglob(os.path.join(vid_path, "*.jpg"))
    frames = map(imread, frame_paths)
    av_frame = stsci.image.average(frames)
    imsave(os.path.join(args.output_dir, vid_name + ".jpg"), av_frame)


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--frames-dir", type=str, required=True)
arg_parser.add_argument("--output-dir", type=str, required=True)
arg_parser.add_argument("--procs", type=int, required=True)
args = arg_parser.parse_args()

pool = mp.Pool(processes=args.procs)
vid_paths = glob.iglob(os.path.join(args.frames_dir, "*"))
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
pool.map(process_video, vid_paths, CHUNKSIZE)

