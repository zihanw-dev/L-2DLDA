from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from l2dlda_core import L2DLDA, accuracy
from run_experiments import summarize
from run_image_matrix_experiments import (
    add_separable_noise,
    load_image_dataset,
    resize_to_square,
    standardize_images,
)


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results"
TAB_DIR = ROOT / "tables"


VARIANTS = [
    {"group": "Iterations", "setting": "$T=1$", "d1": 12, "d2": 12, "lr": 0.70, "lc": 0.70, "T": 1},
    {"group": "Iterations", "setting": "$T=3$", "d1": 12, "d2": 12, "lr": 0.70, "lc": 0.70, "T": 3},
    {"group": "Iterations", "setting": "$T=8$", "d1": 12, "d2": 12, "lr": 0.70, "lc": 0.70, "T": 8},
    {"group": "Projection", "setting": "$(8,8)$", "d1": 8, "d2": 8, "lr": 0.70, "lc": 0.70, "T": 8},
    {"group": "Projection", "setting": "$(12,12)$", "d1": 12, "d2": 12, "lr": 0.70, "lc": 0.70, "T": 8},
    {"group": "Projection", "setting": "$(16,16)$", "d1": 16, "d2": 16, "lr": 0.70, "lc": 0.70, "T": 8},
    {"group": "Shrinkage", "setting": "$(0.7,0.7)$", "d1": 12, "d2": 12, "lr": 0.70, "lc": 0.70, "T": 8},
    {"group": "Shrinkage", "setting": "$(0.5,0.9)$", "d1": 12, "d2": 12, "lr": 0.50, "lc": 0.90, "T": 8},
    {"group": "Shrinkage", "setting": "$(0.9,0.5)$", "d1": 12, "d2": 12, "lr": 0.90, "lc": 0.50, "T": 8},
]


def run_ablation(reps: int = 50, seed: int = 4242) -> pd.DataFrame:
    x, y = load_image_dataset("Digits")
    x = standardize_images(resize_to_square(x, target=32))
    rng = np.random.default_rng(seed)
    rows = []
    shots = 5
    test_per_class = 50
    for rep in tqdm(range(reps), desc="ablation Digits-32 sigma=1.2"):
        train_idx = []
        test_idx = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            picked = rng.choice(idx, size=shots + test_per_class, replace=False)
            train_idx.extend(picked[:shots])
            test_idx.extend(picked[shots:])
        x_train = add_separable_noise(x[train_idx], rng, scale=1.2)
        x_test = add_separable_noise(x[test_idx], rng, scale=1.2)
        y_train, y_test = y[train_idx], y[test_idx]
        for cfg in VARIANTS:
            model = L2DLDA(
                d1=cfg["d1"],
                d2=cfg["d2"],
                lambda_shrink=0.70,
                lambda_row=cfg["lr"],
                lambda_col=cfg["lc"],
                metric_alpha=1.0,
                max_iter=cfg["T"],
            ).fit(x_train, y_train)
            rows.append(
                {
                    "rep": rep,
                    "group": cfg["group"],
                    "setting": cfg["setting"],
                    "accuracy": accuracy(model.predict(x_test), y_test),
                }
            )
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame) -> None:
    summ = summarize(df, ["group", "setting"], "accuracy")
    order = [(v["group"], v["setting"]) for v in VARIANTS]
    with open(TAB_DIR / "hyperparameter_ablation_table.tex", "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{llc}\n")
        f.write("\\toprule\n")
        f.write("Factor & Setting & Accuracy \\\\\n")
        f.write("\\midrule\n")
        for group, setting in order:
            row = summ[(summ["group"] == group) & (summ["setting"] == setting)].iloc[0]
            val = f"{100 * row['mean']:.2f}$\\pm${100 * row['ci95']:.2f}"
            f.write(f"{group} & {setting} & {val} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def main() -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = run_ablation()
    df.to_csv(RES_DIR / "hyperparameter_ablation.csv", index=False)
    write_table(df)
    print(summarize(df, ["group", "setting"], "accuracy").to_string())


if __name__ == "__main__":
    main()
