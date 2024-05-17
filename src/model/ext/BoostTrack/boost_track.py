"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""

from __future__ import print_function

from copy import deepcopy
from typing import List, Optional

import numpy as np

from .assoc import associate, iou_batch
from .ecc import ECC
from .kalmanfilter import KalmanFilter


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,h,r] where x,y is the centre of the box and h is the height and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0

    r = w / float(h + 1e-6)

    return np.array([x, y, h, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,h,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """

    h = x[2]
    r = x[3]
    w = 0 if r <= 0 else r * h

    if score is None:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]
        ).reshape((1, 4))
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox, emb: Optional[np.ndarray] = None):
        """
        Initialises a tracker using initial bounding box.

        """

        self.bbox_to_z_func = convert_bbox_to_z
        self.x_to_bbox_func = convert_x_to_bbox

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = KalmanFilter(self.bbox_to_z_func(bbox))
        self.emb = emb
        self.hit_streak = 0
        self.age = 0

    def get_confidence(self, coef: float = 0.9) -> float:
        n = 7

        if self.age < n:
            return coef ** (n - self.age)
        return coef ** (self.time_since_update - 1)

    def update(self, bbox: np.ndarray, score: float = 0):
        """
        Updates the state vector with observed bbox.
        """

        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.bbox_to_z_func(bbox), score)

    def camera_update(self, transform: np.ndarray):
        x1, y1, x2, y2 = self.get_state()[0]
        x1_, y1_, _ = transform @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = transform @ np.array([x2, y2, 1]).T
        w, h = x2_ - x1_, y2_ - y1_
        cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:4] = [cx, cy, h, w / h]

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """

        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.x_to_bbox_func(self.kf.x)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.x_to_bbox_func(self.kf.x)

    def update_emb(self, emb, alpha=0.9):
        self.emb = alpha * self.emb + (1 - alpha) * emb
        self.emb /= np.linalg.norm(self.emb)

    def get_emb(self):
        return self.emb


class BoostTrack(object):
    def __init__(
        self,
        det_thresh: float,
        lambda_iou: float = 0.5,
        lambda_mhd: float = 0.25,
        lambda_shape: float = 0.25,
        use_dlo_boost: bool = True,
        use_duo_boost: bool = True,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        use_ecc: bool = False,
        dlo_boost_coef: float = 0.65,
        video_name: Optional[str] = None,
    ):

        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.min_hits = min_hits
        self.dlo_boost_coef = dlo_boost_coef

        self.lambda_iou = lambda_iou
        self.lambda_mhd = lambda_mhd
        self.lambda_shape = lambda_shape

        self.use_dlo_boost = use_dlo_boost
        self.use_duo_boost = use_duo_boost
        self.use_ecc = use_ecc
        if use_ecc:
            self.ecc = ECC(scale=350, video_name=video_name, use_cache=False)
        else:
            self.ecc = None

    def update(self, dets, img_tensor, img_numpy, tag):
        if dets is None:
            return np.empty((0, 5))
        if not isinstance(dets, np.ndarray):
            dets = dets.cpu().detach().numpy()

        self.frame_count += 1
        mahalanobis_distance = None

        # Rescale
        scale = min(
            img_tensor.shape[2] / img_numpy.shape[0],
            img_tensor.shape[3] / img_numpy.shape[1],
        )
        dets = deepcopy(dets)
        dets[:, :4] /= scale

        if self.use_ecc:
            transform = self.ecc(img_numpy, self.frame_count, tag)
            for trk in self.trackers:
                trk.camera_update(transform)

        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        confs = np.zeros((len(self.trackers), 1))

        for t in range(len(trks)):
            pos = self.trackers[t].predict()[0]
            confs[t] = self.trackers[t].get_confidence()
            trks[t] = [pos[0], pos[1], pos[2], pos[3], confs[t, 0]]

        if self.use_dlo_boost:
            dets = self.do_iou_confidence_boost(dets)

        if self.use_duo_boost:
            dets = self.do_mh_dist_confidence_boost(dets)

        remain_inds = dets[:, 4] >= self.det_thresh
        dets = dets[remain_inds]
        scores = dets[:, 4]

        if mahalanobis_distance is not None and mahalanobis_distance.size > 0:
            mahalanobis_distance = mahalanobis_distance[remain_inds]
        else:
            mahalanobis_distance = self.get_mh_dist_matrix(dets)

        # Generate embeddings
        dets_embs = np.ones((dets.shape[0], 1))
        emb_cost = None

        matched, unmatched_dets, unmatched_trks = associate(
            dets,
            trks,
            self.iou_threshold,
            mahalanobis_distance=mahalanobis_distance,
            track_confidence=confs,
            detection_confidence=scores,
            emb_cost=emb_cost,
            lambda_iou=self.lambda_iou,
            lambda_mhd=self.lambda_mhd,
            lambda_shape=self.lambda_shape,
        )

        trust = (dets[:, 4] - self.det_thresh) / (1 - self.det_thresh)
        af = 0.95
        dets_alpha = af + (1 - af) * (1 - trust)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], scores[m[0]])
            self.trackers[m[1]].update_emb(dets_embs[m[0]], alpha=dets_alpha[m[0]])

        for i in unmatched_dets:
            if dets[i, 4] >= self.det_thresh:
                self.trackers.append(KalmanBoxTracker(dets[i, :], emb=dets_embs[i]))

        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                # +1 as MOT benchmark requires positive
                ret.append(
                    np.concatenate((d, [trk.id + 1], [trk.get_confidence()])).reshape(
                        1, -1
                    )
                )
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    def dump_cache(self):
        if self.use_ecc:
            self.ecc.save_cache()

    def get_iou_matrix(self, detections: np.ndarray) -> np.ndarray:
        trackers = np.zeros((len(self.trackers), 5))
        for t, trk in enumerate(trackers):
            pos = self.trackers[t].get_state()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        return iou_batch(detections, trackers)

    def get_mh_dist_matrix(self, detections: np.ndarray, n_dims: int = 4) -> np.ndarray:
        if len(self.trackers) == 0:
            return np.zeros((0, 0))
        z = np.zeros((len(detections), n_dims), dtype=float)
        x = np.zeros((len(self.trackers), n_dims), dtype=float)
        sigma_inv = np.zeros_like(x, dtype=float)

        f = self.trackers[0].bbox_to_z_func
        for i in range(len(detections)):
            z[i, :n_dims] = f(detections[i, :]).reshape((-1,))[:n_dims]
        for i in range(len(self.trackers)):
            x[i] = self.trackers[i].kf.x[:n_dims]
            # Note: we assume diagonal covariance matrix
            sigma_inv[i] = np.reciprocal(
                np.diag(self.trackers[i].kf.covariance[:n_dims, :n_dims])
            )

        return (
            (z.reshape((-1, 1, n_dims)) - x.reshape((1, -1, n_dims))) ** 2
            * sigma_inv.reshape((1, -1, n_dims))
        ).sum(axis=2)

    def do_mh_dist_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        n_dims = 4
        limit = 13.2767
        mahalanobis_distance = self.get_mh_dist_matrix(detections, n_dims)

        if mahalanobis_distance.size > 0 and self.frame_count > 1:
            min_mh_dists = mahalanobis_distance.min(1)

            mask = (min_mh_dists > limit) & (detections[:, 4] < self.det_thresh)
            boost_detections = detections[mask]
            boost_detections_args = np.argwhere(mask).reshape((-1,))
            iou_limit = 0.3
            if len(boost_detections) > 0:
                bdiou = iou_batch(boost_detections, boost_detections) - np.eye(
                    len(boost_detections)
                )
                bdiou_max = bdiou.max(axis=1)

                remaining_boxes = boost_detections_args[bdiou_max <= iou_limit]
                args = np.argwhere(bdiou_max > iou_limit).reshape((-1,))
                for i in range(len(args)):
                    boxi = args[i]
                    tmp = np.argwhere(bdiou[boxi] > iou_limit).reshape((-1,))
                    args_tmp = np.append(
                        np.intersect1d(
                            boost_detections_args[args], boost_detections_args[tmp]
                        ),
                        boost_detections_args[boxi],
                    )

                    conf_max = np.max(detections[args_tmp, 4])
                    if detections[boost_detections_args[boxi], 4] == conf_max:
                        remaining_boxes = np.array(
                            remaining_boxes.tolist() + [boost_detections_args[boxi]]
                        )

                mask = np.zeros_like(detections[:, 4], dtype=np.bool_)
                mask[remaining_boxes] = True

            detections[:, 4] = np.where(mask, self.det_thresh + 1e-4, detections[:, 4])
        return detections

    def do_iou_confidence_boost(self, detections: np.ndarray) -> np.ndarray:
        iou_matrix = self.get_iou_matrix(detections)
        ids = np.zeros(len(detections), dtype=np.bool_)

        if iou_matrix.size > 0 and self.frame_count > 1:

            max_iou = iou_matrix.max(1)
            coef = self.dlo_boost_coef
            ids[
                (detections[:, 4] < self.det_thresh)
                & (max_iou * coef >= self.det_thresh)
            ] = True
            detections[:, 4] = np.maximum(detections[:, 4], max_iou * coef)

        return detections
