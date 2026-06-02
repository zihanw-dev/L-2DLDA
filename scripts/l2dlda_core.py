from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def sym(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def scaled_identity(s: np.ndarray) -> np.ndarray:
    scale = float(np.trace(s) / max(s.shape[0], 1))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    return scale * np.eye(s.shape[0])


def normalize_pair(sigma_r: np.ndarray, sigma_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fix the Kronecker scale ambiguity by enforcing tr(Sigma_r) / r = 1."""
    scale = float(np.trace(sigma_r) / max(sigma_r.shape[0], 1))
    if not np.isfinite(scale) or scale <= 1e-12:
        return sigma_r, sigma_c
    return sigma_r / scale, sigma_c * scale


def reg_inv(s: np.ndarray, jitter: float = 1e-4) -> np.ndarray:
    s = sym(s)
    scale = max(float(np.trace(s) / max(s.shape[0], 1)), 1.0)
    return np.linalg.pinv(s + jitter * scale * np.eye(s.shape[0]))


def top_gen_vecs(between: np.ndarray, within: np.ndarray, dim: int, jitter: float) -> np.ndarray:
    between = sym(between)
    within = sym(within)
    scale = max(float(np.trace(within) / max(within.shape[0], 1)), 1.0)
    within = within + jitter * scale * np.eye(within.shape[0])
    dim = min(dim, within.shape[0])
    vals, vecs = eigh(between, within)
    order = np.argsort(vals)[::-1]
    return np.asarray(vecs[:, order[:dim]], dtype=np.float64)


@dataclass
class L2DLDA:
    d1: int = 8
    d2: int = 8
    lambda_shrink: float = 0.35
    lambda_row: Optional[float] = None
    lambda_col: Optional[float] = None
    max_iter: int = 8
    jitter: float = 1e-4
    track: bool = False
    init: str = "identity"
    projected_metric: bool = True
    metric_alpha: float = 1.0

    classes_: Optional[np.ndarray] = None
    means_: Optional[np.ndarray] = None
    U_: Optional[np.ndarray] = None
    V_: Optional[np.ndarray] = None
    sigma_r_: Optional[np.ndarray] = None
    sigma_c_: Optional[np.ndarray] = None
    sigma_r_inv_: Optional[np.ndarray] = None
    sigma_c_inv_: Optional[np.ndarray] = None
    Wr_: Optional[np.ndarray] = None
    Wc_: Optional[np.ndarray] = None
    history_: Optional[list[dict]] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "L2DLDA":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        if x.ndim != 3:
            raise ValueError(f"x must have shape (n,r,c), got {x.shape}")
        n, r, c = x.shape
        classes = np.unique(y)
        means = np.stack([x[y == cls].mean(axis=0) for cls in classes])
        global_mean = x.mean(axis=0)
        cmap = {cls: i for i, cls in enumerate(classes)}
        resid = x - np.stack([means[cmap[cls]] for cls in y])

        if self.init == "random":
            rng = np.random.default_rng(12345)
            a = rng.standard_normal((r, r))
            b = rng.standard_normal((c, c))
            sigma_r = a @ a.T / max(r, 1) + np.eye(r)
            sigma_c = b @ b.T / max(c, 1) + np.eye(c)
            sigma_r, sigma_c = normalize_pair(sigma_r, sigma_c)
        else:
            sigma_r = np.eye(r)
            sigma_c = np.eye(c)
        lambda_r = self.lambda_shrink if self.lambda_row is None else self.lambda_row
        lambda_c = self.lambda_shrink if self.lambda_col is None else self.lambda_col
        history = []
        for it in range(self.max_iter):
            old_r = sigma_r.copy()
            old_c = sigma_c.copy()
            inv_c = reg_inv(sigma_c, self.jitter)
            s_r = sum(e @ inv_c @ e.T for e in resid) / max(n * c, 1)
            sigma_r = (1 - lambda_r) * s_r + lambda_r * scaled_identity(s_r)

            inv_r = reg_inv(sigma_r, self.jitter)
            s_c = sum(e.T @ inv_r @ e for e in resid) / max(n * r, 1)
            sigma_c = (1 - lambda_c) * s_c + lambda_c * scaled_identity(s_c)
            sigma_r, sigma_c = normalize_pair(sigma_r, sigma_c)

            if self.track:
                dr = np.linalg.norm(sigma_r - old_r, "fro") / (np.linalg.norm(old_r, "fro") + 1e-12)
                dc = np.linalg.norm(sigma_c - old_c, "fro") / (np.linalg.norm(old_c, "fro") + 1e-12)
                history.append({"iter": it + 1, "delta_r": float(dr), "delta_c": float(dc)})

        inv_r = reg_inv(sigma_r, self.jitter)
        inv_c = reg_inv(sigma_c, self.jitter)

        srb = np.zeros((r, r))
        scb = np.zeros((c, c))
        for j, cls in enumerate(classes):
            nk = int(np.sum(y == cls))
            diff = means[j] - global_mean
            srb += nk * (diff @ diff.T) / max(c, 1)
            scb += nk * (diff.T @ diff) / max(r, 1)

        U = top_gen_vecs(srb, sigma_r, self.d1, self.jitter)
        V = top_gen_vecs(scb, sigma_c, self.d2, self.jitter)

        self.classes_ = classes
        self.means_ = means
        self.U_ = U
        self.V_ = V
        self.sigma_r_ = sigma_r
        self.sigma_c_ = sigma_c
        self.sigma_r_inv_ = inv_r
        self.sigma_c_inv_ = inv_c
        if self.projected_metric:
            self.Wr_ = reg_inv(U.T @ sigma_r @ U, self.jitter)
            self.Wc_ = reg_inv(V.T @ sigma_c @ V, self.jitter)
        else:
            self.Wr_ = sym(U.T @ inv_r @ U)
            self.Wc_ = sym(V.T @ inv_c @ V)
        self.history_ = history
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        scores = np.zeros((x.shape[0], len(self.classes_)))
        for i, sample in enumerate(x):
            for j, mean in enumerate(self.means_):
                d = self.U_.T @ (sample - mean) @ self.V_
                mahal = float(np.trace(self.Wc_ @ d.T @ self.Wr_ @ d))
                euclid = float(np.sum(d * d))
                scores[i, j] = -(
                    self.metric_alpha * mahal
                    + (1.0 - self.metric_alpha) * euclid
                )
        return self.classes_[np.argmax(scores, axis=1)]

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        z = np.asarray([self.U_.T @ sample @ self.V_ for sample in x])
        return z.reshape(z.shape[0], -1)


@dataclass
class L2DLDALinearProbe:
    d1: int = 8
    d2: int = 8
    lambda_shrink: float = 0.35
    max_iter: int = 8
    jitter: float = 1e-4

    feature_model_: Optional[L2DLDA] = None
    clf_: Optional[object] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "L2DLDALinearProbe":
        feature_model = L2DLDA(
            d1=self.d1,
            d2=self.d2,
            lambda_shrink=self.lambda_shrink,
            max_iter=self.max_iter,
            jitter=self.jitter,
        ).fit(x, y)
        z = feature_model.transform(x)
        clf = make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )
        clf.fit(z, y)
        self.feature_model_ = feature_model
        self.clf_ = clf
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = self.feature_model_.transform(x)
        return self.clf_.predict(z)


@dataclass
class TwoDLDA:
    d1: int = 8
    d2: int = 8
    jitter: float = 1e-4

    classes_: Optional[np.ndarray] = None
    means_: Optional[np.ndarray] = None
    U_: Optional[np.ndarray] = None
    V_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TwoDLDA":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        n, r, c = x.shape
        classes = np.unique(y)
        means = np.stack([x[y == cls].mean(axis=0) for cls in classes])
        global_mean = x.mean(axis=0)
        srb = np.zeros((r, r))
        scb = np.zeros((c, c))
        for j, cls in enumerate(classes):
            diff = means[j] - global_mean
            nk = int(np.sum(y == cls))
            srb += nk * (diff @ diff.T) / max(c, 1)
            scb += nk * (diff.T @ diff) / max(r, 1)
        # This baseline mirrors the Euclidean 2D-LDA variant commonly used in
        # small-sample matrix classification: it uses between-class scatter to
        # select row/column subspaces and Euclidean nearest-centroid scoring.
        self.U_ = top_gen_vecs(srb, np.eye(r), self.d1, self.jitter)
        self.V_ = top_gen_vecs(scb, np.eye(c), self.d2, self.jitter)
        self.classes_ = classes
        self.means_ = means
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = []
        mean_proj = [self.U_.T @ m @ self.V_ for m in self.means_]
        for sample in np.asarray(x, dtype=np.float64):
            z = self.U_.T @ sample @ self.V_
            dist = [np.sum((z - mp) ** 2) for mp in mean_proj]
            preds.append(self.classes_[int(np.argmin(dist))])
        return np.asarray(preds)


@dataclass
class MPCA2D:
    d1: int = 8
    d2: int = 8
    jitter: float = 1e-4

    classes_: Optional[np.ndarray] = None
    means_: Optional[np.ndarray] = None
    U_: Optional[np.ndarray] = None
    V_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MPCA2D":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        _, r, c = x.shape
        centered = x - x.mean(axis=0, keepdims=True)
        row_cov = sum(sample @ sample.T for sample in centered) / max(x.shape[0] * c, 1)
        col_cov = sum(sample.T @ sample for sample in centered) / max(x.shape[0] * r, 1)
        self.U_ = top_gen_vecs(row_cov, np.eye(r), self.d1, self.jitter)
        self.V_ = top_gen_vecs(col_cov, np.eye(c), self.d2, self.jitter)
        self.classes_ = np.unique(y)
        self.means_ = np.stack([x[y == cls].mean(axis=0) for cls in self.classes_])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = []
        mean_proj = [self.U_.T @ m @ self.V_ for m in self.means_]
        for sample in np.asarray(x, dtype=np.float64):
            z = self.U_.T @ sample @ self.V_
            dist = [np.sum((z - mp) ** 2) for mp in mean_proj]
            preds.append(self.classes_[int(np.argmin(dist))])
        return np.asarray(preds)


@dataclass
class DATER2D:
    d1: int = 8
    d2: int = 8
    max_iter: int = 5
    jitter: float = 1e-4
    within_shrink: float = 0.85

    classes_: Optional[np.ndarray] = None
    means_: Optional[np.ndarray] = None
    U_: Optional[np.ndarray] = None
    V_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DATER2D":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        _, r, c = x.shape
        classes = np.unique(y)
        means = np.stack([x[y == cls].mean(axis=0) for cls in classes])
        global_mean = x.mean(axis=0)
        self.V_ = MPCA2D(d1=self.d1, d2=self.d2, jitter=self.jitter).fit(x, y).V_
        self.U_ = np.eye(r, min(self.d1, r))

        for _ in range(self.max_iter):
            sw_r = np.zeros((r, r))
            sb_r = np.zeros((r, r))
            for j, cls in enumerate(classes):
                mask = y == cls
                mv = means[j] @ self.V_
                gv = global_mean @ self.V_
                sb_r += int(np.sum(mask)) * ((mv - gv) @ (mv - gv).T) / max(self.d2, 1)
                for sample in x[mask]:
                    diff = (sample - means[j]) @ self.V_
                    sw_r += diff @ diff.T / max(self.d2, 1)
            sw_r = (1 - self.within_shrink) * sw_r + self.within_shrink * scaled_identity(sw_r)
            self.U_ = top_gen_vecs(sb_r, sw_r, self.d1, self.jitter)

            sw_c = np.zeros((c, c))
            sb_c = np.zeros((c, c))
            for j, cls in enumerate(classes):
                mask = y == cls
                mu = self.U_.T @ means[j]
                gu = self.U_.T @ global_mean
                sb_c += int(np.sum(mask)) * ((mu - gu).T @ (mu - gu)) / max(self.d1, 1)
                for sample in x[mask]:
                    diff = self.U_.T @ (sample - means[j])
                    sw_c += diff.T @ diff / max(self.d1, 1)
            sw_c = (1 - self.within_shrink) * sw_c + self.within_shrink * scaled_identity(sw_c)
            self.V_ = top_gen_vecs(sb_c, sw_c, self.d2, self.jitter)

        self.classes_ = classes
        self.means_ = means
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = []
        mean_proj = [self.U_.T @ m @ self.V_ for m in self.means_]
        for sample in np.asarray(x, dtype=np.float64):
            z = self.U_.T @ sample @ self.V_
            dist = [np.sum((z - mp) ** 2) for mp in mean_proj]
            preds.append(self.classes_[int(np.argmin(dist))])
        return np.asarray(preds)


def nearest_centroid_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    train_v = x_train.reshape(x_train.shape[0], -1)
    test_v = x_test.reshape(x_test.shape[0], -1)
    centers = np.stack([train_v[y_train == cls].mean(axis=0) for cls in classes])
    dists = ((test_v[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    return classes[np.argmin(dists, axis=1)]


def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred) == np.asarray(target)))
