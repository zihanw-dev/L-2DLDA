from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from l2dlda_core import L2DLDA, TwoDLDA, accuracy, nearest_centroid_predict


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"
RES_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})


def finalize_figure(fig, ax):
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)
    fig.tight_layout()


def save_figure(fig, stem: str):
    fig.savefig(FIG_DIR / f"{stem}.pdf", facecolor="white", edgecolor="white", transparent=False)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, facecolor="white", edgecolor="white", transparent=False)


def ar1(dim: int, rho: float) -> np.ndarray:
    idx = np.arange(dim)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def sample_matrix_normal(rng, mean, sigma_r, sigma_c, n):
    r, c = mean.shape
    lr = np.linalg.cholesky(sigma_r + 1e-8 * np.eye(r))
    lc = np.linalg.cholesky(sigma_c + 1e-8 * np.eye(c))
    z = rng.standard_normal((n, r, c))
    return np.asarray([mean + lr @ zi @ lc.T for zi in z], dtype=np.float64)


def make_low_rank_means(rng, r: int, c: int, rank: int = 2, gap: float = 1.8):
    ur, _ = np.linalg.qr(rng.standard_normal((r, rank)))
    vc, _ = np.linalg.qr(rng.standard_normal((c, rank)))
    a = np.diag(np.linspace(gap, 0.6 * gap, rank))
    m1 = ur @ a @ vc.T
    m2 = -m1
    return m1, m2


def ridge_accuracy(x_train, y_train, x_test, y_test):
    train_v = x_train.reshape(x_train.shape[0], -1)
    test_v = x_test.reshape(x_test.shape[0], -1)
    clf = make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
    clf.fit(train_v, y_train)
    return accuracy(clf.predict(test_v), y_test)


def shrinkage_lda_accuracy(x_train, y_train, x_test, y_test):
    train_v = x_train.reshape(x_train.shape[0], -1)
    test_v = x_test.reshape(x_test.shape[0], -1)
    clf = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    )
    clf.fit(train_v, y_train)
    return accuracy(clf.predict(test_v), y_test)


def synthetic_dimension(seed: int, reps: int, dims: list[int]):
    rng = np.random.default_rng(seed)
    rows = []
    for dim in tqdm(dims, desc="synthetic dimension"):
        for rep in range(reps):
            local = np.random.default_rng(rng.integers(1_000_000_000))
            r = c = dim
            sigma_r = ar1(r, 0.65)
            sigma_c = ar1(c, 0.65)
            m1, m2 = make_low_rank_means(local, r, c, rank=2, gap=2.4)
            n_train = 8
            n_test = 120
            x0 = sample_matrix_normal(local, m1, sigma_r, sigma_c, n_train)
            x1 = sample_matrix_normal(local, m2, sigma_r, sigma_c, n_train)
            xt0 = sample_matrix_normal(local, m1, sigma_r, sigma_c, n_test)
            xt1 = sample_matrix_normal(local, m2, sigma_r, sigma_c, n_test)
            x_train = np.concatenate([x0, x1])
            y_train = np.asarray([0] * n_train + [1] * n_train)
            x_test = np.concatenate([xt0, xt1])
            y_test = np.asarray([0] * n_test + [1] * n_test)

            ours = L2DLDA(d1=2, d2=2, lambda_shrink=0.10, max_iter=8).fit(x_train, y_train)
            base = TwoDLDA(d1=2, d2=2).fit(x_train, y_train)
            rows.extend(
                [
                    {
                        "experiment": "dimension",
                        "dim": dim,
                        "features": dim * dim,
                        "rep": rep,
                        "method": "L-2DLDA",
                        "error": 1 - accuracy(ours.predict(x_test), y_test),
                    },
                    {
                        "experiment": "dimension",
                        "dim": dim,
                        "features": dim * dim,
                        "rep": rep,
                        "method": "2DLDA",
                        "error": 1 - accuracy(base.predict(x_test), y_test),
                    },
                    {
                        "experiment": "dimension",
                        "dim": dim,
                        "features": dim * dim,
                        "rep": rep,
                        "method": "Nearest centroid",
                        "error": 1 - accuracy(nearest_centroid_predict(x_train, y_train, x_test), y_test),
                    },
                    {
                        "experiment": "dimension",
                        "dim": dim,
                        "features": dim * dim,
                        "rep": rep,
                        "method": "Ridge",
                        "error": 1 - ridge_accuracy(x_train, y_train, x_test, y_test),
                    },
                ]
            )
    return pd.DataFrame(rows)


def synthetic_sample_size(seed: int, reps: int):
    rng = np.random.default_rng(seed + 11)
    rows = []
    r = c = 60
    sigma_r = ar1(r, 0.65)
    sigma_c = ar1(c, 0.65)
    n_tests = 160
    for n_train in tqdm([3, 5, 8, 12, 20, 30], desc="synthetic sample size"):
        for rep in range(reps):
            local = np.random.default_rng(rng.integers(1_000_000_000))
            m1, m2 = make_low_rank_means(local, r, c, rank=2, gap=2.4)
            x0 = sample_matrix_normal(local, m1, sigma_r, sigma_c, n_train)
            x1 = sample_matrix_normal(local, m2, sigma_r, sigma_c, n_train)
            xt0 = sample_matrix_normal(local, m1, sigma_r, sigma_c, n_tests)
            xt1 = sample_matrix_normal(local, m2, sigma_r, sigma_c, n_tests)
            x_train = np.concatenate([x0, x1])
            y_train = np.asarray([0] * n_train + [1] * n_train)
            x_test = np.concatenate([xt0, xt1])
            y_test = np.asarray([0] * n_tests + [1] * n_tests)
            ours = L2DLDA(d1=2, d2=2, lambda_shrink=0.10, max_iter=8).fit(x_train, y_train)
            base = TwoDLDA(d1=2, d2=2).fit(x_train, y_train)
            rows.extend(
                [
                    {"experiment": "sample", "n_train": n_train, "rep": rep, "method": "L-2DLDA", "error": 1 - accuracy(ours.predict(x_test), y_test)},
                    {"experiment": "sample", "n_train": n_train, "rep": rep, "method": "2DLDA", "error": 1 - accuracy(base.predict(x_test), y_test)},
                    {"experiment": "sample", "n_train": n_train, "rep": rep, "method": "Nearest centroid", "error": 1 - accuracy(nearest_centroid_predict(x_train, y_train, x_test), y_test)},
                ]
            )
    return pd.DataFrame(rows)


def flipflop_convergence(seed: int, reps: int):
    rng = np.random.default_rng(seed + 23)
    rows = []
    settings = [(40, 0.3, 0.1), (40, 0.7, 0.5), (80, 0.7, 0.5), (80, 0.9, 0.9)]
    for r, rho, lam in settings:
        c = r
        sigma_r = ar1(r, rho)
        sigma_c = ar1(c, rho)
        for init in ["identity", "random"]:
            for rep in tqdm(range(reps), desc=f"flip-flop r={r} rho={rho} lambda={lam} init={init}"):
                local = np.random.default_rng(rng.integers(1_000_000_000))
                m1, m2 = make_low_rank_means(local, r, c, rank=2, gap=2.4)
                x0 = sample_matrix_normal(local, m1, sigma_r, sigma_c, 8)
                x1 = sample_matrix_normal(local, m2, sigma_r, sigma_c, 8)
                x_train = np.concatenate([x0, x1])
                y_train = np.asarray([0] * 8 + [1] * 8)
                model = L2DLDA(d1=2, d2=2, lambda_shrink=lam, max_iter=20, track=True, init=init).fit(x_train, y_train)
                for item in model.history_:
                    rows.append({
                        "rep": rep,
                        "r": r,
                        "rho": rho,
                        "lambda": lam,
                        "init": init,
                        **item,
                        "delta": 0.5 * (item["delta_r"] + item["delta_c"]),
                    })
    return pd.DataFrame(rows)


def signal_sensitivity(seed: int, reps: int):
    rng = np.random.default_rng(seed + 59)
    rows = []
    r = c = 60
    sigma_r = ar1(r, 0.65)
    sigma_c = ar1(c, 0.65)
    for gap in [0.8, 1.2, 1.6, 2.0, 2.4]:
        for rep in tqdm(range(reps), desc=f"signal={gap}"):
            local = np.random.default_rng(rng.integers(1_000_000_000))
            m1, m2 = make_low_rank_means(local, r, c, rank=2, gap=gap)
            x0 = sample_matrix_normal(local, m1, sigma_r, sigma_c, 8)
            x1 = sample_matrix_normal(local, m2, sigma_r, sigma_c, 8)
            xt0 = sample_matrix_normal(local, m1, sigma_r, sigma_c, 160)
            xt1 = sample_matrix_normal(local, m2, sigma_r, sigma_c, 160)
            x_train = np.concatenate([x0, x1])
            y_train = np.asarray([0] * 8 + [1] * 8)
            x_test = np.concatenate([xt0, xt1])
            y_test = np.asarray([0] * 160 + [1] * 160)
            ours = L2DLDA(d1=2, d2=2, lambda_shrink=0.10, max_iter=8).fit(x_train, y_train)
            base = TwoDLDA(d1=2, d2=2).fit(x_train, y_train)
            rows.append({"gap": gap, "rep": rep, "method": "L-2DLDA", "error": 1 - accuracy(ours.predict(x_test), y_test)})
            rows.append({"gap": gap, "rep": rep, "method": "2DLDA", "error": 1 - accuracy(base.predict(x_test), y_test)})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    out = df.groupby(group_cols)[value_col].agg(["mean", "std", "count"]).reset_index()
    out["ci95"] = 1.96 * out["std"] / np.sqrt(out["count"])
    return out


def plot_dimension(df: pd.DataFrame):
    summ = summarize(df, ["features", "method"], "error")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method, sub in summ.groupby("method"):
        sub = sub.sort_values("features")
        ax.errorbar(sub["features"], sub["mean"], yerr=sub["ci95"], marker="o", linewidth=1.8, label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Vectorized dimension $rc$")
    ax.set_ylabel("Classification error")
    ax.set_title("Dimension scaling under separable structured noise")
    ax.legend(frameon=False)
    finalize_figure(fig, ax)
    save_figure(fig, "synthetic_dimension")


def plot_sample(df: pd.DataFrame):
    summ = summarize(df, ["n_train", "method"], "error")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for method, sub in summ.groupby("method"):
        sub = sub.sort_values("n_train")
        ax.errorbar(sub["n_train"], sub["mean"], yerr=sub["ci95"], marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Training samples per class")
    ax.set_ylabel("Classification error")
    ax.set_title("Sample-size scaling at $r=c=60$")
    ax.legend(frameon=False)
    finalize_figure(fig, ax)
    save_figure(fig, "synthetic_sample_size")


def plot_convergence(df: pd.DataFrame):
    summ = summarize(df, ["iter", "r", "rho", "lambda", "init"], "delta")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for (r, rho, lam, init), sub in summ.groupby(["r", "rho", "lambda", "init"]):
        if init == "random" and r == 40:
            continue
        label = f"r={r}, rho={rho}, lambda={lam}, {init}"
        ax.plot(sub["iter"], sub["mean"], marker="o", linewidth=1.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Flip-flop iteration")
    ax.set_ylabel("Relative covariance change")
    ax.set_title("Regularized flip-flop iterations stabilize rapidly")
    ax.legend(frameon=False, fontsize=7)
    finalize_figure(fig, ax)
    save_figure(fig, "flipflop_convergence")


def plot_signal(df: pd.DataFrame):
    summ = summarize(df, ["gap", "method"], "error")
    fig, ax = plt.subplots(figsize=(5.7, 3.6))
    for method, sub in summ.groupby("method"):
        ax.errorbar(sub["gap"], sub["mean"], yerr=sub["ci95"], marker="o", linewidth=1.8, label=method)
    ax.set_xlabel("Signal amplitude")
    ax.set_ylabel("Classification error")
    ax.set_title("Synthetic sensitivity to signal strength")
    ax.legend(frameon=False)
    finalize_figure(fig, ax)
    save_figure(fig, "signal_sensitivity")


def write_latex_tables(dim_df, sample_df):
    dim = summarize(dim_df, ["features", "method"], "error")
    pivot = dim.pivot(index="features", columns="method", values="mean")
    with open(TAB_DIR / "dimension_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rcccc}\n\\toprule\n$rc$ & L-2DLDA & 2DLDA & Nearest centroid & Ridge \\\\\n\\midrule\n")
        for idx, row in pivot.sort_index().iterrows():
            f.write(f"{int(idx)} & {100*row.get('L-2DLDA', np.nan):.2f} & {100*row.get('2DLDA', np.nan):.2f} & {100*row.get('Nearest centroid', np.nan):.2f} & {100*row.get('Ridge', np.nan):.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    for path in [FIG_DIR, TAB_DIR, RES_DIR, DATA_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    dims = [10, 20, 40, 80] if args.fast else [10, 20, 40, 80, 120]
    reps = 4 if args.fast else args.reps
    dim_df = synthetic_dimension(args.seed, reps, dims)
    sample_df = synthetic_sample_size(args.seed, reps)
    conv_df = flipflop_convergence(args.seed, max(4, reps))
    signal_df = signal_sensitivity(args.seed, reps)

    dim_df.to_csv(RES_DIR / "synthetic_dimension.csv", index=False)
    sample_df.to_csv(RES_DIR / "synthetic_sample_size.csv", index=False)
    conv_df.to_csv(RES_DIR / "flipflop_convergence.csv", index=False)
    signal_df.to_csv(RES_DIR / "signal_sensitivity.csv", index=False)
    with open(RES_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "reps": reps, "dims": dims}, f, indent=2)

    plot_dimension(dim_df)
    plot_sample(sample_df)
    plot_convergence(conv_df)
    plot_signal(signal_df)
    write_latex_tables(dim_df, sample_df)


if __name__ == "__main__":
    main()
