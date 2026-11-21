"""Evaluation metrics for landmark-based registration."""
import cv2
import numpy as np

from .landmarks import OUTER_EYE_CORNERS


def landmark_error(pts, ref_pts):
    """Per-landmark Euclidean distance (pixels)."""
    return np.linalg.norm(np.asarray(pts) - np.asarray(ref_pts), axis=1)


def nme(pts, ref_pts):
    """Mean landmark error normalised by the reference inter-ocular distance."""
    i, j = OUTER_EYE_CORNERS
    iod = np.linalg.norm(ref_pts[i] - ref_pts[j])
    return landmark_error(pts, ref_pts).mean() / iod


def face_mask(ref_pts, size):
    """Binary mask of the convex hull of the reference landmarks."""
    mask = np.zeros((size, size), np.uint8)
    hull = cv2.convexHull(np.round(ref_pts).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def standardise_gray(img, mask=None):
    """Grayscale, zero mean / unit std inside the mask (removes global lighting)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) if img.ndim == 3 else img.astype(float)
    m = mask if mask is not None else np.ones_like(g, bool)
    return (g - g[m].mean()) / (g[m].std() + 1e-8)


def difference_image(img, ref_img, mask=None):
    """|standardised(img) - standardised(ref)|."""
    return np.abs(standardise_gray(img, mask) - standardise_gray(ref_img, mask))


def masked_mad(img, ref_img, mask):
    """Mean absolute (standardised) intensity difference inside the face mask."""
    return float(difference_image(img, ref_img, mask)[mask].mean())
