"""Facial landmark detection.

Uses the MediaPipe FaceLandmarker (478-point face mesh) and maps it to the
standard 68-point layout (iBUG 300-W / dlib / OpenFace), so the rest of the
pipeline works with the same landmark convention that OpenFace uses.
"""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "face_landmarker.task"

# MediaPipe face-mesh index for each of the 68 iBUG points (0-based).
# Order follows the iBUG convention: points go from the image-left side
# (subject's right) to the image-right side.
MP_TO_68 = np.array([
    # 0-16 jaw line
    127, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 356,
    # 17-21 right eyebrow (image left), 22-26 left eyebrow
    70, 63, 105, 66, 107,
    336, 296, 334, 293, 300,
    # 27-30 nose bridge, 31-35 nose base
    168, 197, 5, 4,
    75, 97, 2, 326, 305,
    # 36-41 right eye, 42-47 left eye
    33, 160, 158, 133, 153, 144,
    362, 385, 387, 263, 373, 380,
    # 48-59 outer lips, 60-67 inner lips
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
])
assert len(MP_TO_68) == 68

# Facial regions in the 68-point layout (used for subsets and for drawing).
REGIONS = {
    "jaw": list(range(0, 17)),
    "right_brow": list(range(17, 22)),
    "left_brow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_base": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lips": list(range(48, 60)),
    "inner_lips": list(range(60, 68)),
}
CLOSED_REGIONS = {"right_eye", "left_eye", "outer_lips", "inner_lips"}

# Outer eye corners: used to normalise errors (inter-ocular distance).
OUTER_EYE_CORNERS = (36, 45)


def head_pose_from_matrix(m):
    """Yaw, pitch, roll in degrees from MediaPipe's 4x4 facial transform."""
    r = np.asarray(m)[:3, :3]
    sy = np.hypot(r[0, 0], r[1, 0])
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    yaw = np.degrees(np.arctan2(-r[2, 0], sy))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


class LandmarkDetector:
    """Detects the 68 facial landmarks of the most central face in an image."""

    def __init__(self, model_path=DEFAULT_MODEL, max_faces=3):
        from mediapipe.tasks.python import BaseOptions, vision
        import mediapipe as mp

        self._mp = mp
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)

    def detect(self, image_bgr, return_all=False):
        """Return a (68, 2) array of (x, y) pixel coordinates, or None.

        If return_all=True, returns a dict with the 68 points, the full
        478-point mesh and the head pose (yaw, pitch, roll in degrees).
        """
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(rgb))
        res = self._lm.detect(mp_img)
        if not res.face_landmarks:
            return None
        # several faces -> keep the one closest to the image centre
        best, best_d = 0, np.inf
        meshes = []
        for i, face in enumerate(res.face_landmarks):
            pts = np.array([[p.x * w, p.y * h] for p in face])
            meshes.append(pts)
            d = np.linalg.norm(pts.mean(0) - [w / 2, h / 2])
            if d < best_d:
                best, best_d = i, d
        mesh = meshes[best]
        pts68 = mesh[MP_TO_68]
        if not return_all:
            return pts68
        pose = (head_pose_from_matrix(res.facial_transformation_matrixes[best])
                if res.facial_transformation_matrixes else (np.nan,) * 3)
        return {"pts": pts68, "mesh": mesh, "yaw": pose[0], "pitch": pose[1],
                "roll": pose[2], "n_faces": len(meshes)}

    def close(self):
        self._lm.close()
