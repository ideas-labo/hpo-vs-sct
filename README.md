# RQ Analysis Experiment Code

This repository contains the code and lightweight analysis artifacts for the tuning experiments used in the paper.

The full benchmark problems and raw experiment outputs are intentionally not tracked in Git because they are large and contain many generated CSV files. Place those artifacts under `data/` and `results/` after downloading them from the external archive for the paper.

## Repository Contents

- `optimizer/`: implementations of the general, SCT, and HPO optimizers.
- `experiments/`: scripts for pre-experiments, budget computation, and main experiments.
- `analysis/`: scripts and lightweight processed tables for the research-question analysis.
- `data/README.md`: placeholder describing where to put external problems.

Not included in Git:

- `data/`: full benchmark problems, about 314 MB in the original workspace.
- `results/`: raw pre/main experiment outputs, about 1.7 GB and more than 90,000 CSV files.
- local metadata such as `.codex/`, `.agents/`, and `.git/`.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some analysis scripts use `rpy2`, which also requires a working R installation. Install the R packages used by your local analysis scripts before running those parts.

## Data Layout

After downloading the external problem archive, the directory should look like this:

```text
data/
  HPO_problems/
    multi-fidelity/
    single-fidelity/
  SCT_problems/
```

The experiment scripts expect paths such as:

```text
data/HPO_problems/multi-fidelity
```

## Running Experiments

Run commands from the repository root.

Pre-experiment:

```bash
PYTHONPATH=. python experiments/pre_experiment/run_pre.py
```

Compute budgets:

```bash
python experiments/pre_experiment/compute_budget.py
```

Main experiment:

```bash
PYTHONPATH=. python experiments/main_experiment/run_main.py
```

The scripts write generated outputs to `result/` or local CSV files. Those generated outputs are ignored by Git.

## Analysis

The `analysis/` directory contains scripts and processed tables for effectiveness, efficiency, landscape, and recommendation analysis. Run the relevant script from the repository root or from its own directory if it depends on local relative paths.

## Archiving Data and Results

For public release, put the full `data/` and `results/` directories in a research artifact archive such as Zenodo, OSF, Figshare, or a GitHub Release asset. Include the archive DOI or download URL here once available.
