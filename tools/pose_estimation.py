#!/usr/bin/env python
import argparse
import os
import sys
import warnings
from glob import glob

import torch
from tqdm import tqdm

sys.path.append(".")
from src import PoseEstimation
from src.utils import json_handler, vis
from src.utils.video import Capture, Writer

warnings.filterwarnings("ignore")


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

    # prepairing output data dirs
    out_dirs = []
    for video_path in video_paths:
        name = os.path.basename(video_path).replace(".mp4", "")
        out_dir = os.path.join(video_dir, name)
        out_dirs.append(out_dir)
        os.makedirs(out_dir, exist_ok=True)

    # load model
    pe = PoseEstimation(args.cfg_path, args.gpu)

    for video_path, out_dir in zip(tqdm(video_paths, ncols=100), out_dirs):

        # load video
        cap = Capture(video_path)

        # predict
        results = []
        if args.video:
            out_frames = []
        for frame_num in tqdm(range(cap.frame_count), ncols=100, leave=False):
            frame = cap.read()[1]
            preds = pe.predict(frame, frame_num)
            results.append(preds)

            if args.video:
                out_frames.append(vis.write_frame(frame))

        # save results
        data_path = os.path.join(out_dir, "json", "pose.json")
        json_handler.dump(results, data_path)
        data_path = os.path.join(out_dir, "json", "frame_shape.json")
        json_handler.dump(cap.size, data_path)

        if args.video:
            video_num = os.path.basename(video_path).split(".")[0]
            out_video_path = os.path.join(out_dir, f"{video_num}_pose.mp4")
            wtr = Writer(out_video_path, cap.fps, cap.size)
            wtr.write_each(out_frames)

        pe.reset_tracker()

        del cap
        if args.video:
            del wtr, out_frames


if __name__ == "__main__":
    main()
