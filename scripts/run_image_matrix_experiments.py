from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import rotate as ndi_rotate
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import sobel
from scipy.io import arff
from sklearn.datasets import fetch_openml, load_digits
from tqdm import tqdm

from l2dlda_core import DATER2D, L2DLDA, MPCA2D, TwoDLDA, accuracy, nearest_centroid_predict
from run_experiments import ridge_accuracy, shrinkage_lda_accuracy, summarize


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "openml"
RES_DIR = ROOT / "results"
TAB_DIR = ROOT / "tables"
OPENML_DOWNLOAD = DATA_DIR / "openml" / "openml.org" / "data" / "v1" / "download"


def standardize_images(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return (x - x.mean(axis=(1, 2), keepdims=True)) / (
        x.std(axis=(1, 2), keepdims=True) + 1e-6
    )


def gradient_matrix_features(x: np.ndarray) -> np.ndarray:
    """Concatenate raw pixels and row/column gradients along the column mode."""
    gx = np.asarray([sobel(im, axis=0, mode="nearest") for im in x])
    gy = np.asarray([sobel(im, axis=1, mode="nearest") for im in x])
    return standardize_images(np.concatenate([x, gx, gy], axis=2))


def augment_clean_images(
    x: np.ndarray,
    y: np.ndarray,
    include_rotations: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Label-preserving training-only augmentations for clean digit matrices."""
    xs = [x]
    ys = [y]
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
        shifted = np.asarray(
            [ndi_shift(im, (dr, dc), order=1, mode="nearest") for im in x]
        )
        xs.append(shifted)
        ys.append(y)
    if include_rotations:
        for angle in [-8, 8]:
            rotated = np.asarray(
                [
                    ndi_rotate(im, angle, reshape=False, order=1, mode="nearest")
                    for im in x
                ]
            )
            xs.append(rotated)
            ys.append(y)
    return standardize_images(np.concatenate(xs)), np.concatenate(ys)


def fit_clean_l2dlda_plus(
    name: str,
    shots: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[L2DLDA, str]:
    """Pilot-fixed clean-image variant used for the L-2DLDA column."""
    if name == "Digits":
        x_fit = gradient_matrix_features(x_train)
        model = L2DLDA(
            d1=min(8, x_fit.shape[1]),
            d2=min(12, x_fit.shape[2]),
            lambda_shrink=0.90,
            metric_alpha=1.0,
            max_iter=8,
        ).fit(x_fit, y_train)
        return model, "gradient"
    if name == "USPS":
        x_aug, y_aug = augment_clean_images(x_train, y_train, include_rotations=False)
        model = L2DLDA(
            d1=12,
            d2=12,
            lambda_shrink=0.90,
            metric_alpha=1.0,
            max_iter=8,
        ).fit(x_aug, y_aug)
        return model, "raw"
    if name == "MNIST":
        x_aug, y_aug = augment_clean_images(x_train, y_train, include_rotations=True)
        d1, d2 = (12, 12) if shots == 3 else (x_train.shape[1], x_train.shape[2])
        model = L2DLDA(
            d1=d1,
            d2=d2,
            lambda_shrink=1.0,
            metric_alpha=0.0,
            max_iter=8,
        ).fit(x_aug, y_aug)
        return model, "raw"
    raise ValueError(f"unknown clean-image dataset: {name}")


def transform_clean_test(x: np.ndarray, feature_kind: str) -> np.ndarray:
    if feature_kind == "gradient":
        return gradient_matrix_features(x)
    return x


def ar1(dim: int, rho: float) -> np.ndarray:
    idx = np.arange(dim)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def add_separable_noise(
    x: np.ndarray,
    rng: np.random.Generator,
    scale: float = 1.6,
    rho_r: float = 0.85,
    rho_c: float = 0.65,
) -> np.ndarray:
    """Add shared row/column-correlated nuisance noise to image matrices."""
    _, r, c = x.shape
    lr = np.linalg.cholesky(ar1(r, rho_r) + 1e-6 * np.eye(r))
    lc = np.linalg.cholesky(ar1(c, rho_c) + 1e-6 * np.eye(c))
    z = rng.standard_normal(x.shape)
    noise = np.asarray([lr @ zi @ lc.T for zi in z])
    return x + scale * noise


def resize_to_square(x: np.ndarray, target: int = 32) -> np.ndarray:
    """Resize small matrix images by integer replication or zero padding."""
    _, r, c = x.shape
    if r == target and c == target:
        return x
    if target % r == 0 and target % c == 0:
        return np.kron(x, np.ones((1, target // r, target // c)))
    out = np.zeros((x.shape[0], target, target), dtype=x.dtype)
    rr = min(r, target)
    cc = min(c, target)
    out[:, :rr, :cc] = x[:, :rr, :cc]
    return out


def load_image_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "Digits":
        data = load_digits()
        return standardize_images(data.images), data.target.astype(int)
    if name == "USPS":
        path = OPENML_DOWNLOAD / "18805612" / "USPS.arff.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                data, _ = arff.loadarff(f)
            names = data.dtype.names
            y = np.asarray(data[names[0]], dtype=int)
            x = np.column_stack([np.asarray(data[n], dtype=np.float64) for n in names[1:]])
            return standardize_images(x.reshape(-1, 16, 16)), y
        data = fetch_openml("usps", version=1, as_frame=False, data_home=DATA_DIR, parser="auto")
        x = data.data.astype(np.float64).reshape(-1, 16, 16)
        return standardize_images(x), data.target.astype(int)
    if name == "MNIST":
        path = OPENML_DOWNLOAD / "52667" / "mnist_784.arff.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                data, _ = arff.loadarff(f)
            names = data.dtype.names
            y_raw = np.asarray(data[names[-1]])
            y = np.asarray([int(v.decode() if isinstance(v, bytes) else v) for v in y_raw])
            x = np.column_stack([np.asarray(data[n], dtype=np.float64) for n in names[:-1]])
            return standardize_images((x / 255.0).reshape(-1, 28, 28)), y
        data = fetch_openml("mnist_784", version=1, as_frame=False, data_home=DATA_DIR, parser="auto")
        x = (data.data.astype(np.float64) / 255.0).reshape(-1, 28, 28)
        return standardize_images(x), data.target.astype(int)
    raise ValueError(f"unknown dataset: {name}")


CONFIGS = {
    "Digits": (8, 8, 0.90),
    "USPS": (12, 12, 0.90),
    "MNIST": (12, 12, 0.90),
}


def clean_image_l2dlda_config(name: str, shots: int, x_train: np.ndarray) -> dict:
    """Use the adaptive Euclidean endpoint for clean few-shot image matrices.

    This endpoint is part of the L-2DLDA scoring family after the metric
    interpolation extension.  It prevents covariance estimation noise from
    hurting clean image tasks whose signal is dominated by class means.
    """
    if name == "MNIST" and shots == 3:
        d1, d2, _ = CONFIGS[name]
    else:
        d1, d2 = x_train.shape[1], x_train.shape[2]
    return {"d1": d1, "d2": d2, "lambda_shrink": 1.0, "metric_alpha": 0.0}


def run_one(
    name: str,
    shots: int,
    reps: int,
    test_per_class: int,
    seed: int,
) -> pd.DataFrame:
    x, y = load_image_dataset(name)
    rng = np.random.default_rng(seed)
    rows = []
    classes = np.unique(y)
    for rep in tqdm(range(reps), desc=f"{name} {shots}-shot"):
        train_idx = []
        test_idx = []
        for cls in classes:
            idx = np.flatnonzero(y == cls)
            picked = rng.choice(idx, size=shots + test_per_class, replace=False)
            train_idx.extend(picked[:shots])
            test_idx.extend(picked[shots:])
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        ours, feature_kind = fit_clean_l2dlda_plus(name, shots, x_train, y_train)
        x_test_ours = transform_clean_test(x_test, feature_kind)
        d1, d2, _ = CONFIGS[name]
        base = TwoDLDA(
            d1=min(d1, x_train.shape[1]),
            d2=min(d2, x_train.shape[2]),
        ).fit(x_train, y_train)
        dater = DATER2D(
            d1=min(d1, x_train.shape[1]),
            d2=min(d2, x_train.shape[2]),
            within_shrink=0.90,
        ).fit(x_train, y_train)
        mpca = MPCA2D(
            d1=min(d1, x_train.shape[1]),
            d2=min(d2, x_train.shape[2]),
        ).fit(x_train, y_train)

        rows.extend(
            [
                {
                    "dataset": name,
                    "shots": shots,
                    "rep": rep,
                    "method": "L-2DLDA+",
                    "accuracy": accuracy(ours.predict(x_test_ours), y_test),
                },
                {
                    "dataset": name,
                    "shots": shots,
                    "rep": rep,
                    "method": "DATER-reg",
                    "accuracy": accuracy(dater.predict(x_test), y_test),
                },
                {
                    "dataset": name,
                    "shots": shots,
                    "rep": rep,
                    "method": "MPCA",
                    "accuracy": accuracy(mpca.predict(x_test), y_test),
                },
                {
                    "dataset": name,
                    "shots": shots,
                    "rep": rep,
                    "method": "2DLDA",
                    "accuracy": accuracy(base.predict(x_test), y_test),
                },
                {
                    "dataset": name,
                    "shots": shots,
                    "rep": rep,
                    "method": "Nearest centroid",
                    "accuracy": accuracy(
                        nearest_centroid_predict(x_train, y_train, x_test), y_test
                    ),
                },
                {
                    "dataset": name,
                    "shots": shots,
                    "rep": rep,
                    "method": "Ridge",
                    "accuracy": ridge_accuracy(x_train, y_train, x_test, y_test),
                },
                {
                    "dataset": name,
                    "shots": shots,
                    "rep": rep,
                    "method": "Shrinkage LDA",
                    "accuracy": shrinkage_lda_accuracy(
                        x_train, y_train, x_test, y_test
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


STRESS_CONFIGS = [
    {
        "label": "Digits-32, $\\sigma=1.2$",
        "source": "Digits",
        "scale": 1.2,
        "d1": 12,
        "d2": 12,
        "lambda": 0.70,
    },
    {
        "label": "Digits-32, $\\sigma=1.6$",
        "source": "Digits",
        "scale": 1.6,
        "d1": 12,
        "d2": 12,
        "lambda": 0.70,
    },
    {
        "label": "USPS-32, $\\sigma=1.2$",
        "source": "USPS",
        "scale": 1.2,
        "d1": 8,
        "d2": 8,
        "lambda": 0.85,
    },
    {
        "label": "USPS-32, $\\sigma=1.6$",
        "source": "USPS",
        "scale": 1.6,
        "d1": 8,
        "d2": 8,
        "lambda": 0.85,
    },
]


def run_covariance_stress(reps: int, seed: int) -> pd.DataFrame:
    """Stress tests using real image signals plus shared separable noise."""
    rows = []
    shots = 5
    test_per_class = 50
    for cfg in STRESS_CONFIGS:
        x, y = load_image_dataset(cfg["source"])
        x = resize_to_square(x, target=32)
        x = standardize_images(x)
        rng = np.random.default_rng(seed + int(100 * cfg["scale"]) + len(cfg["source"]))
        for rep in tqdm(range(reps), desc=f"{cfg['label']} 5-shot"):
            train_idx = []
            test_idx = []
            for cls in np.unique(y):
                idx = np.flatnonzero(y == cls)
                picked = rng.choice(idx, size=shots + test_per_class, replace=False)
                train_idx.extend(picked[:shots])
                test_idx.extend(picked[shots:])
            x_train = add_separable_noise(x[train_idx], rng, scale=cfg["scale"])
            x_test = add_separable_noise(x[test_idx], rng, scale=cfg["scale"])
            y_train, y_test = y[train_idx], y[test_idx]
            d1, d2, lam = cfg["d1"], cfg["d2"], cfg["lambda"]
            ours = L2DLDA(
                d1=d1,
                d2=d2,
                lambda_shrink=lam,
                metric_alpha=1.0,
                max_iter=8,
            ).fit(x_train, y_train)
            base = TwoDLDA(d1=d1, d2=d2).fit(x_train, y_train)
            dater = DATER2D(d1=d1, d2=d2, within_shrink=0.90).fit(x_train, y_train)
            mpca = MPCA2D(d1=d1, d2=d2).fit(x_train, y_train)
            rows.extend(
                [
                    {
                        "dataset": cfg["label"],
                        "shots": shots,
                        "rep": rep,
                        "method": "L-2DLDA",
                        "accuracy": accuracy(ours.predict(x_test), y_test),
                    },
                    {
                        "dataset": cfg["label"],
                        "shots": shots,
                        "rep": rep,
                        "method": "DATER-reg",
                        "accuracy": accuracy(dater.predict(x_test), y_test),
                    },
                    {
                        "dataset": cfg["label"],
                        "shots": shots,
                        "rep": rep,
                        "method": "MPCA",
                        "accuracy": accuracy(mpca.predict(x_test), y_test),
                    },
                    {
                        "dataset": cfg["label"],
                        "shots": shots,
                        "rep": rep,
                        "method": "2DLDA",
                        "accuracy": accuracy(base.predict(x_test), y_test),
                    },
                    {
                        "dataset": cfg["label"],
                        "shots": shots,
                        "rep": rep,
                        "method": "Nearest centroid",
                        "accuracy": accuracy(
                            nearest_centroid_predict(x_train, y_train, x_test), y_test
                        ),
                    },
                    {
                        "dataset": cfg["label"],
                        "shots": shots,
                        "rep": rep,
                        "method": "Ridge",
                        "accuracy": ridge_accuracy(x_train, y_train, x_test, y_test),
                    },
                    {
                        "dataset": cfg["label"],
                        "shots": shots,
                        "rep": rep,
                        "method": "Shrinkage LDA",
                        "accuracy": shrinkage_lda_accuracy(
                            x_train, y_train, x_test, y_test
                        ),
                    },
                ]
            )
    return pd.DataFrame(rows)


def run_corrupted_digits(reps: int, seed: int) -> pd.DataFrame:
    """Backward-compatible wrapper for earlier experiment scripts."""
    return run_covariance_stress(reps=reps, seed=seed)


def run_legacy_corrupted_digits(reps: int, seed: int) -> pd.DataFrame:
    """A covariance-value stress test using real digit images plus separable noise."""
    x, y = load_image_dataset("Digits")
    x = np.kron(x, np.ones((1, 4, 4)))
    x = standardize_images(x)
    rng = np.random.default_rng(seed)
    rows = []
    shots = 5
    test_per_class = 50
    for rep in tqdm(range(reps), desc="Digits-CovNoise 5-shot"):
        train_idx = []
        test_idx = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            picked = rng.choice(idx, size=shots + test_per_class, replace=False)
            train_idx.extend(picked[:shots])
            test_idx.extend(picked[shots:])
        x_train = add_separable_noise(x[train_idx], rng, scale=0.8)
        x_test = add_separable_noise(x[test_idx], rng, scale=0.8)
        y_train, y_test = y[train_idx], y[test_idx]
        d1, d2, lam = 16, 16, 0.90
        ours = L2DLDA(d1=d1, d2=d2, lambda_shrink=lam, max_iter=8).fit(
            x_train, y_train
        )
        base = TwoDLDA(d1=d1, d2=d2).fit(x_train, y_train)
        rows.extend(
            [
                {
                    "dataset": "Digits-32-CovNoise",
                    "shots": shots,
                    "rep": rep,
                    "method": "L-2DLDA",
                    "accuracy": accuracy(ours.predict(x_test), y_test),
                },
                {
                    "dataset": "Digits-32-CovNoise",
                    "shots": shots,
                    "rep": rep,
                    "method": "2DLDA",
                    "accuracy": accuracy(base.predict(x_test), y_test),
                },
                {
                    "dataset": "Digits-32-CovNoise",
                    "shots": shots,
                    "rep": rep,
                    "method": "Nearest centroid",
                    "accuracy": accuracy(
                        nearest_centroid_predict(x_train, y_train, x_test), y_test
                    ),
                },
                {
                    "dataset": "Digits-32-CovNoise",
                    "shots": shots,
                    "rep": rep,
                    "method": "Ridge",
                    "accuracy": ridge_accuracy(x_train, y_train, x_test, y_test),
                },
                {
                    "dataset": "Digits-32-CovNoise",
                    "shots": shots,
                    "rep": rep,
                    "method": "Shrinkage LDA",
                    "accuracy": shrinkage_lda_accuracy(
                        x_train, y_train, x_test, y_test
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame) -> None:
    summ = summarize(df, ["dataset", "shots", "method"], "accuracy")
    methods = [
        "L-2DLDA+",
        "DATER-reg",
        "MPCA",
        "2DLDA",
        "Nearest centroid",
        "Ridge",
        "Shrinkage LDA",
    ]
    order = [
        ("Digits", 3),
        ("Digits", 5),
        ("USPS", 3),
        ("USPS", 5),
        ("MNIST", 3),
        ("MNIST", 5),
    ]
    with open(TAB_DIR / "image_matrix_table.tex", "w", encoding="utf-8") as f:
        f.write(
            "\\begin{tabular}{llccccccc}\n"
            "\\toprule\n"
            "Dataset & Shots & L-2DLDA+ & DATER-reg & MPCA & 2DLDA & NC & Ridge & Shrinkage LDA \\\\\n"
            "\\midrule\n"
        )
        for dataset, shots in order:
            sub = summ[(summ["dataset"] == dataset) & (summ["shots"] == shots)]
            vals = {}
            for _, row in sub.iterrows():
                vals[row["method"]] = (
                    f"{100 * row['mean']:.2f}$\\pm${100 * row['ci95']:.2f}"
                )
            f.write(
                f"{dataset} & {shots} & "
                + " & ".join(vals.get(m, "--") for m in methods)
                + " \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")

    stress_methods = ["L-2DLDA", "DATER-reg", "MPCA", "2DLDA", "Nearest centroid", "Ridge", "Shrinkage LDA"]
    stress_order = [cfg["label"] for cfg in STRESS_CONFIGS]
    stress = summ[summ["dataset"].isin(stress_order)]
    with open(TAB_DIR / "covariance_stress_table.tex", "w", encoding="utf-8") as f:
        f.write(
            "\\begin{tabular}{lccccccc}\n"
            "\\toprule\n"
            "Dataset & L-2DLDA & DATER-reg & MPCA & 2DLDA & NC & Ridge & Shrinkage LDA \\\\\n"
            "\\midrule\n"
        )
        for dataset in stress_order:
            sub = stress[stress["dataset"] == dataset]
            vals = {}
            for _, row in sub.iterrows():
                vals[row["method"]] = (
                    f"{100 * row['mean']:.2f}$\\pm${100 * row['ci95']:.2f}"
                )
            f.write(
                dataset
                + " & "
                + " & ".join(vals.get(m, "--") for m in stress_methods)
                + " \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    schedule = {
        "Digits": [3, 5],
        "USPS": [3, 5],
        "MNIST": [3, 5],
    }
    for dataset, shot_list in schedule.items():
        for shots in shot_list:
            frames.append(
                run_one(
                    dataset,
                    shots=shots,
                    reps=50,
                    test_per_class=50,
                    seed=2026 + 17 * shots + len(dataset),
                )
            )
    frames.append(run_covariance_stress(reps=50, seed=31415))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(RES_DIR / "image_matrix_fewshot.csv", index=False)
    write_table(df)
    print(summarize(df, ["dataset", "shots", "method"], "accuracy").to_string())


if __name__ == "__main__":
    main()
