from dataclasses import dataclass


@dataclass(frozen=True)
class Format:
    n_frame: str = "n_frame"
    id: str = "id"
    bbox: str = "bbox"
    keypoints: str = "keypoints"
