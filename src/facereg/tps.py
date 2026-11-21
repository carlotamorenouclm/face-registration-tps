"""Thin-plate spline (TPS) transformation with regularisation.

f(x) = a0 + A x + sum_i w_i U(|x - c_i|),   U(r) = r^2 log r

Fitting solves
    [ K + lam*I   P ] [w]   [v]
    [ P^T         0 ] [a] = [0]
where K_ij = U(|c_i - c_j|) and P = [1, x, y]. With lam = 0 the spline
interpolates the control points exactly; lam > 0 trades exactness for a
smoother (lower bending energy) deformation. Points are internally
normalised to unit scale so lam does not depend on the image size.
"""
import cv2
import numpy as np

from .shape import apply_affine, similarity_transform


def _U(r2):
    """TPS kernel written in terms of squared distance: r^2 log r = 0.5 r^2 log r^2."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 0.5 * r2 * np.log(r2)
    out[r2 == 0] = 0.0
    return out


class ThinPlateSpline:
    def __init__(self, lam=0.0):
        self.lam = float(lam)

    def fit(self, src, dst):
        """Estimate f such that f(src_i) ~ dst_i. src, dst: (K, 2)."""
        src = np.asarray(src, float)
        dst = np.asarray(dst, float)
        self._mu = src.mean(0)
        self._s = np.sqrt(((src - self._mu) ** 2).sum(1).mean())
        c = (src - self._mu) / self._s
        n = len(c)
        K = _U(((c[:, None] - c[None]) ** 2).sum(-1))
        P = np.hstack([np.ones((n, 1)), c])
        L = np.zeros((n + 3, n + 3))
        L[:n, :n] = K + self.lam * np.eye(n)
        L[:n, n:] = P
        L[n:, :n] = P.T
        rhs = np.zeros((n + 3, 2))
        rhs[:n] = dst
        sol = np.linalg.solve(L, rhs)
        self._c, self.w, self.a = c, sol[:n], sol[n:]
        return self

    def transform(self, pts):
        pts = (np.asarray(pts, float) - self._mu) / self._s
        out = np.empty((len(pts), 2))
        # chunk to keep memory bounded on big grids
        for i in range(0, len(pts), 65536):
            p = pts[i:i + 65536]
            K = _U(((p[:, None] - self._c[None]) ** 2).sum(-1))
            out[i:i + 65536] = K @ self.w + self.a[0] + p @ self.a[1:]
        return out

    def bending_energy(self):
        """w^T K w (summed over x and y): 0 for a purely affine map."""
        K = _U(((self._c[:, None] - self._c[None]) ** 2).sum(-1))
        return float(np.trace(self.w.T @ K @ self.w))


def border_anchors(size, n_per_side=3):
    """Points along the border of a size x size canvas."""
    t = np.linspace(0, size - 1, n_per_side + 1)[:-1]
    e = size - 1
    return np.vstack([np.c_[t, np.zeros_like(t)], np.c_[np.full_like(t, e), t],
                      np.c_[e - t, np.full_like(t, e)], np.c_[np.zeros_like(t), e - t]])


def tps_warp(image, moving_pts, ref_pts, out_size, lam=0.0, anchors=True,
             return_tps=False):
    """Warp `image` so that `moving_pts` land on `ref_pts`.

    Backward mapping: a TPS f: reference canvas -> moving image is fitted on
    the landmark pairs and evaluated on every output pixel; the image is then
    resampled with bilinear interpolation (cv2.remap).

    anchors=True adds points on the canvas border, mapped into the moving
    image with the best similarity transform, so the background follows a
    global similarity instead of extrapolating the spline wildly.
    """
    ref_pts = np.asarray(ref_pts, float)
    moving_pts = np.asarray(moving_pts, float)
    src, dst = ref_pts, moving_pts
    if anchors:
        A = similarity_transform(ref_pts, moving_pts)
        b = border_anchors(out_size)
        src = np.vstack([ref_pts, b])
        dst = np.vstack([moving_pts, apply_affine(A, b)])
    tps = ThinPlateSpline(lam).fit(src, dst)
    yy, xx = np.mgrid[0:out_size, 0:out_size]
    grid = np.c_[xx.ravel(), yy.ravel()].astype(float)
    m = tps.transform(grid).astype(np.float32)
    map_x = m[:, 0].reshape(out_size, out_size)
    map_y = m[:, 1].reshape(out_size, out_size)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if return_tps:
        return warped, tps, (map_x, map_y)
    return warped


def similarity_warp(image, moving_pts, ref_pts, out_size):
    """Baseline: global similarity (Procrustes) alignment only."""
    A = similarity_transform(moving_pts, ref_pts)
    return cv2.warpAffine(image, A, (out_size, out_size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0), A
