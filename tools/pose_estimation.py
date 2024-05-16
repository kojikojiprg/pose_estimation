#!/usr/bin/env python
import argparse
import os
import sys
import warnings
from glob import glob

import torch

sys.path.append(".")
from src import PoseEstimation
from src.visualize import Visualizer

warnings.simplefilter("ignore")


def parser():
    parser = argparse.ArgumentParser()

    # requires
    parser.add_argument(
        "video_dir",
        type=str,
        help="path of video directory",
    )

    # options
    parser.add_argument("-c", "--cfg_path", type=str, default="configs/config.yaml")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu number")
    parser.add_argument(
        "-v", "--video", default=False, action="store_true", help="with writing video"
    )

    args = parser.parse_args()

    return args


def main():
    args = parser()

    torch.cuda.set_device(args.gpu)

    # load video paths
    video_dir = args.video_dir
    video_paths = sorted(glob(os.path.join(video_dir, "*.mp4")))
    print(f"=> video paths:\n{video_paths}")

    # prepairing output data dirs
    out_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        out_dir = os.path.join(video_dir, name)
        out_dirs.append(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # load model
    pe = PoseEstimation(args.cfg_path, args.gpu)

    if args.video:
        vis = Visualizer(args)

    for video_path, out_dir in zip(video_paths, out_dirs):
        print(f"=> processing {video_path}")
        pe.inference(video_path, out_dir)

        if args.video:
            vis.visualise(video_path, out_dir)


if __name__ == "__main__":
    main()
