"""Landmark shapes: similarity alignment, Procrustes and the mean shape."""
import numpy as np


def similarity_transform(src, dst):
    """Least-squares similarity (scale, rotation, translation) mapping src->dst.

    Umeyama's method. Returns a 2x3 matrix A such that dst ~ A @ [x, y, 1].
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = d0.T @ s0 / len(src)
    u, sig, vt = np.linalg.svd(cov)
    d = np.sign(np.linalg.det(u @ vt))
    D = np.diag([1.0, d])
    R = u @ D @ vt
    scale = np.trace(np.diag(sig) @ D) / s0.var(0).sum()
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]])


def apply_affine(A, pts):
    pts = np.asarray(pts, float)
    return pts @ A[:, :2].T + A[:, 2]


def _normalise(shape):
    """Centre at the origin and scale to unit centroid size."""
    c = shape - shape.mean(0)
    return c / np.sqrt((c ** 2).sum())


def generalized_procrustes(shapes, n_iter=20, tol=1e-8):
    """Generalized Procrustes Analysis.

    shapes: (N, K, 2). Returns (mean_shape, aligned_shapes), both normalised
    (centroid at 0, unit centroid size).
    """
    shapes = np.asarray(shapes, float)
    aligned = np.array([_normalise(s) for s in shapes])
    mean = aligned[0].copy()
    for _ in range(n_iter):
        for i, s in enumerate(aligned):
            aligned[i] = apply_affine(similarity_transform(s, mean), s)
        new_mean = _normalise(aligned.mean(0))
        # keep the mean's orientation fixed to the first estimate
        new_mean = apply_affine(similarity_transform(new_mean, mean), new_mean)
        new_mean = _normalise(new_mean)
        if np.abs(new_mean - mean).max() < tol:
            mean = new_mean
            break
        mean = new_mean
    return mean, aligned


def place_in_canvas(shape, size=256, face_width=0.56, center_y=0.56,
                    eye_corners=(36, 45)):
    """Scale/rotate a normalised shape into a size x size pixel canvas.

    The shape is rotated so the eyes are horizontal, scaled so the jaw width
    is `face_width` * size, and centred horizontally, with the landmark
    bounding-box centre at `center_y` * size.
    """
    s = np.asarray(shape, float)
    v = s[eye_corners[1]] - s[eye_corners[0]]
    ang = -np.arctan2(v[1], v[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    s = s @ R.T
    width = s[:, 0].max() - s[:, 0].min()
    s = s * (face_width * size / width)
    lo, hi = s.min(0), s.max(0)
    s = s - (lo + hi) / 2 + [size / 2, center_y * size]
    return s
