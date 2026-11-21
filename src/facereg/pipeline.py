"""High-level steps: detect a dataset, build the average face, register."""
from pathlib import Path

import cv2
import numpy as np

from . import metrics
from .shape import generalized_procrustes, place_in_canvas
from .tps import similarity_warp, tps_warp


def detect_folder(detector, folder, pattern="*.jpg"):
    """Detect landmarks in every image of a folder.

    Returns a list of dicts: name, img, pts (68x2), yaw, pitch, roll.
    Images where no face is found are reported and skipped.
    """
    out = []
    for f in sorted(Path(folder).glob(pattern)):
        img = cv2.imread(str(f))
        r = detector.detect(img, return_all=True)
        if r is None:
            print(f"no face found: {f.name}")
            continue
        out.append({"name": f.stem, "img": img, **r})
    return out


def select_frontal(faces, max_yaw=15.0, max_pitch=15.0):
    return [f for f in faces if abs(f["yaw"]) <= max_yaw and abs(f["pitch"]) <= max_pitch]


def build_average_face(faces, size=256, lam=0.0):
    """Average landmark shape (GPA) and average image (TPS-warped mean).

    Returns (mean_pts in canvas coords, mean_image uint8, warped list).
    """
    shapes = np.array([f["pts"] for f in faces])
    mean_norm, _ = generalized_procrustes(shapes)
    mean_pts = place_in_canvas(mean_norm, size)
    warped = [tps_warp(f["img"], f["pts"], mean_pts, size, lam=lam) for f in faces]
    mean_img = np.mean(np.array(warped, float), axis=0).round().astype(np.uint8)
    return mean_pts, mean_img, warped


def register(detector, img, pts, mean_pts, mean_img, size=256, lam=0.0, idx=None):
    """Register one image to the average face and evaluate it.

    Three stages are compared, each measured by re-detecting the landmarks
    in the produced image (so the 'after' error is not trivially zero):
      none        - image just resized to the canvas
      similarity  - global Procrustes alignment (translation/rotation/scale)
      tps         - thin-plate spline on the landmarks in `idx` (all if None)
    """
    idx = np.arange(68) if idx is None else np.asarray(idx)
    h, w = img.shape[:2]
    none_img = cv2.resize(img, (size, size))
    sim_img, _ = similarity_warp(img, pts, mean_pts, size)
    tps_img, tps, maps = tps_warp(img, pts[idx], mean_pts[idx], size, lam=lam,
                                  return_tps=True)
    mask = metrics.face_mask(mean_pts, size)
    res = {"images": {"none": none_img, "similarity": sim_img, "tps": tps_img},
           "tps": tps, "maps": maps, "mask": mask, "pts": {}, "nme": {},
           "err": {}, "mad": {}}
    for k, im in res["images"].items():
        p = detector.detect(im)
        res["pts"][k] = p
        if p is None:
            res["nme"][k] = np.nan
            continue
        res["err"][k] = metrics.landmark_error(p, mean_pts)
        res["nme"][k] = metrics.nme(p, mean_pts)
        res["mad"][k] = metrics.masked_mad(im, mean_img, mask)
    return res
