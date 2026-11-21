"""Plotting helpers."""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from .landmarks import CLOSED_REGIONS, REGIONS


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img


def draw_contours(ax, pts, color="lime", lw=1.2, **kw):
    """Draw the facial contours (jaw, brows, nose, eyes, lips) of a 68-pt shape."""
    for name, idx in REGIONS.items():
        p = pts[idx]
        if name in CLOSED_REGIONS:
            p = np.vstack([p, p[:1]])
        ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=lw, **kw)


def show_landmarks(ax, img, pts, title=None, numbers=False, contours=True):
    ax.imshow(rgb(img))
    if contours:
        draw_contours(ax, pts, color="yellow", lw=0.8, alpha=0.8)
    ax.scatter(pts[:, 0], pts[:, 1], s=6, c="red")
    if numbers:
        for i, (x, y) in enumerate(pts):
            ax.text(x + 1, y - 1, str(i), fontsize=5, color="cyan")
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def show_grid(ax, map_x, map_y, step=16, img=None, color="w"):
    """Visualise a backward deformation: where each output grid line samples
    the moving image. Drawn in moving-image coordinates."""
    if img is not None:
        ax.imshow(rgb(img))
    for i in range(0, map_x.shape[0], step):
        ax.plot(map_x[i, :], map_y[i, :], color=color, lw=0.6)
    for j in range(0, map_x.shape[1], step):
        ax.plot(map_x[:, j], map_y[:, j], color=color, lw=0.6)
    ax.axis("off")


def overlay(img_a, img_b):
    """Red/cyan overlay of two images (a -> red, b -> green+blue)."""
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b
    return np.dstack([ga, gb, gb])


def savefig(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
