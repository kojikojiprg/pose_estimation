import numpy as np
from numpy.typing import NDArray

from .model import Detector, Tracker
from .utils import yaml_handler
from .utils.format import Format as PoseDataFormat


class PoseEstimation:
    def __init__(
        self,
        cfg_path: str,
        gpu: int,
    ):
        self._device = f"cuda:{gpu}"
        self._cfg = yaml_handler.load(cfg_path)
        self._detector = Detector(self._cfg, self._device)
        self._tracker = Tracker(self._cfg, self._device)

    def __del__(self):
        del self._detector, self._tracker

    def reset_tracker(self):
        del self._tracker
        self._tracker = Tracker(self._cfg, self._device)

    def predict(self, frame: NDArray, frame_num: int):
        # keypoints detection
        bboxs, kps = self._detector.predict(frame)

        # tracking
        tracks = self._tracker.update(bboxs, frame)
        tracks = tracks[np.argsort(tracks[:, 4])]  # sort by track id

        # append result
        results = []
        for t in tracks:
            # get id is closed kps
            i = np.where(np.isclose(t[:4], bboxs[:, :4]))[0]
            if len(i) == 0:
                continue

            # select minimum index
            i = np.min(i)

            # create result
            result = {
                PoseDataFormat.n_frame: int(frame_num),
                PoseDataFormat.id: int(t[4]),
                PoseDataFormat.bbox: bboxs[i],
                PoseDataFormat.keypoints: kps[i],
            }
            results.append(result)

        return results
