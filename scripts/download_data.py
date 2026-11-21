"""Download the data and model needed by the project.

1. MediaPipe FaceLandmarker model (face_landmarker.task, ~3.7 MB) -> models/
2. LFW (Labeled Faces in the Wild, funneled version) via scikit-learn.
   The full archive (~230 MB) is cached in ~/scikit_learn_data (outside the
   repo); a random subset with ONE image per person is copied to data/lfw/.

Usage:
    python scripts/download_data.py            # 60 people (default)
    python scripts/download_data.py --n 100
"""
import argparse
import random
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
LFW_OUT = ROOT / "data" / "lfw"


def download_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"model already present: {MODEL_PATH}")
        return
    print(f"downloading {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"saved {MODEL_PATH} ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")


def download_lfw(n_people, seed):
    # Private helper, but it only downloads/extracts the archive without
    # loading all 13k images into memory (fetch_lfw_people would).
    from sklearn.datasets._lfw import _check_fetch_lfw
    _, folder = _check_fetch_lfw(funneled=True, download_if_missing=True)
    folder = Path(folder)
    people = sorted(p for p in folder.iterdir() if p.is_dir())
    rng = random.Random(seed)
    rng.shuffle(people)
    LFW_OUT.mkdir(parents=True, exist_ok=True)
    copied = 0
    for person in people[:n_people]:
        img = sorted(person.glob("*.jpg"))[0]
        shutil.copy(img, LFW_OUT / img.name)
        copied += 1
    print(f"copied {copied} images (1 per person) to {LFW_OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of people")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    download_model()
    download_lfw(args.n, args.seed)
