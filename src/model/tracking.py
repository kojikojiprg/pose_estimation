import gc
import importlib
import os
import sys
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import yaml
from torchvision.transforms import transforms as T

sys.path.append(os.path.join("submodules/unitrack/"))
import utils

importlib.reload(utils)
sys.path.append("submodules")
from unitrack.tracker.mot.pose import PoseAssociationTracker


class Tracker:
    def __init__(self, cfg: dict, device: str):
        # set config
        opts = SimpleNamespace(**{})
        with open(cfg["configs"]["unitrack"]) as f:
            common_args = yaml.safe_load(f)
        for k, v in common_args["common"].items():
            setattr(opts, k, v)
        for k, v in common_args["posetrack"].items():
            setattr(opts, k, v)
        opts.return_stage = 2
        opts.device = device  # assign device

        self.transforms = T.Compose(
            [T.ToTensor(), T.Normalize(opts.im_mean, opts.im_std)]
        )

        self.tracker = PoseAssociationTracker(opts)

    def __del__(self):
        del self.tracker, self.transforms
        gc.collect()

    def update(self, img: np.array, kps_all: np.array):
        process_img = img.copy()

        # Normalize RGB
        process_img = process_img / 255.0
        process_img = process_img[:, :, ::-1]
        process_img = np.ascontiguousarray(process_img)
        process_img = self.transforms(process_img)

        obs = np.array([self._cvt_kp2ob(kps) for kps in kps_all])

        tracks = self.tracker.update(process_img, img, obs)
        for t in tracks:
            if not isinstance(t.pose[0], list):
                t.pose = self._cvt_ob2kp(t.pose)
            else:
                tracks.remove(t)  # remove not updated track

        return tracks

    @staticmethod
    def _cvt_kp2ob(kps: np.array):
        # https://github.com/leonid-pishchulin/poseval
        return [
            {"id": [i], "x": [kp[0]], "y": [kp[1]], "score": [kp[2]]}
            for i, kp in enumerate(kps)
        ]

    @staticmethod
    def _cvt_ob2kp(ob: List[Dict[str, list]]):
        return [[pt["x"][0], pt["y"][0], pt["score"][0]] for pt in ob]
