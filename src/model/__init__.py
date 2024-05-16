import gc
from types import SimpleNamespace
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from mmpose.structures import PoseDataSample
from tqdm import tqdm

from src.utils.video import Capture

from ..format import Format
from .detection import Detector
from .tracking import Tracker


class PoseModel:
    def __init__(
        self,
        pose_cfg: SimpleNamespace,
        gpu: int,
    ):
        self._cfg = pose_cfg

        print("=> loading detector model")
        self._detector = Detector(self._cfg, f"cuda:{gpu}")
        print("=> loading tracker model")
        self._tracker = Tracker(self._cfg, f"cuda:{gpu}")

    def __del__(self):
        del self._detector, self._tracker
        gc.collect()
        torch.cuda.empty_cache()

    def predict(self, cap: Capture) -> List[Dict[str, Any]]:
        print("=> running pose estimation")
        results = []
        for frame_num in tqdm(range(cap.frame_count), ncols=100):
            frame = cap.read()[1]

            # keypoints detection
            bboxs, kps = self._detector.predict(frame)

            # tracking
            tracks = self._tracker.update(bboxs, frame)
            tracks = tracks[np.argsort(tracks[:, 4])]  # sort by track id

            # append result
            for t in tracks:
                # get id is closed kps
                i = np.where(np.isclose(t[:4], bboxs[:, :4]))[0]
                if len(i) == 0:
                    continue

                # select minimum index
                i = np.min(i)

                # create result
                result = {
                    Format.n_frame: int(frame_num),
                    Format.id: int(t[4]),
                    Format.bbox: bboxs[i],
                    Format.keypoints: kps[i],
                }

                results.append(result)

            del bboxs, kps, tracks
        gc.collect()

        return results
