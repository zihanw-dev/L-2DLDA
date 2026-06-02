# L-2DLDA

This repository contains the code accompanying the manuscript
**"Shrinkage Kronecker covariance for robust two-dimensional discriminant
analysis in small samples"**.

L-2DLDA is a likelihood-based two-dimensional discriminant analysis method for
matrix-valued classification in high-dimensional small-sample settings. The
method estimates shrinkage-regularized Kronecker covariance factors, uses them
for covariance-aware row and column feature extraction, and classifies projected
matrices with a Mahalanobis score.

## Contents

```text
scripts/l2dlda_core.py
    Core implementations of L-2DLDA, Euclidean 2DLDA, MPCA, DATER-reg,
    nearest-centroid utilities, and evaluation helpers.

scripts/run_experiments.py
    Synthetic matrix-normal experiments, sample-size scaling experiments,
    flip-flop diagnostics, and signal-strength sensitivity experiments.

scripts/run_image_matrix_experiments.py
    Clean few-shot image-matrix experiments and covariance-value stress tests
    on public digit datasets.

scripts/run_hyperparameter_ablation.py
    Ablations over flip-flop sweeps, projection dimensions, and row/column
    shrinkage settings.

results/
    CSV files generated from the reported runs.

tables/ and figures/
    LaTeX table fragments and PDF/PNG figures used by the manuscript.
```

The repository intentionally excludes cached datasets and Python cache files.
OpenML datasets are downloaded on first use into `data/openml/`.

## Installation

The experiments were run with Python 3.11 on CPU. Create a clean environment and
install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Check

Run a short synthetic smoke test:

```bash
python scripts/run_experiments.py --fast
```

Expected outputs are written to:

```text
results/*.csv
tables/*.tex
figures/*.pdf
figures/*.png
```

## Reproducing the Reported Experiments

From the repository root, run:

```bash
python scripts/run_experiments.py --reps 50
python scripts/run_image_matrix_experiments.py
python scripts/run_hyperparameter_ablation.py
```

The default scripts use fixed random seeds. The image-matrix script uses public
datasets from scikit-learn and OpenML, including Digits, USPS, and MNIST. The
first OpenML run requires internet access and may take several minutes on CPU.

## Mapping to the Manuscript

```text
Synthetic dimension-scaling figure
    scripts/run_experiments.py -> figures/synthetic_dimension.pdf

Synthetic sample-size figure
    scripts/run_experiments.py -> figures/synthetic_sample_size.pdf

Flip-flop convergence figure
    scripts/run_experiments.py -> figures/flipflop_convergence.pdf

Signal-strength sensitivity figure
    scripts/run_experiments.py -> figures/signal_sensitivity.pdf

Clean few-shot image-matrix table
    scripts/run_image_matrix_experiments.py -> tables/image_matrix_table.tex

Covariance-value stress-test table
    scripts/run_image_matrix_experiments.py -> tables/covariance_stress_table.tex

Hyperparameter ablation table
    scripts/run_hyperparameter_ablation.py -> tables/hyperparameter_ablation_table.tex
```

## Data Availability

The experiments use public benchmark datasets. `sklearn.datasets.load_digits`
provides Digits. `sklearn.datasets.fetch_openml` downloads USPS and MNIST into
the local cache directory `data/openml/` on first use. No private or proprietary
data are required.

## Citation

If you use this code, please cite the manuscript once a bibliographic record is
available. A provisional citation file is provided in `CITATION.cff`.

## License

This code is released under the MIT License. See `LICENSE` for details.
