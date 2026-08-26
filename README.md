# Are Optimizing Model Hyperparameter and Software Configuration Alike?

This repository contains the experiment code, optimizer implementations, analysis scripts, and lightweight processed artifacts for the following paper:

Xiaowei Tang, Yulong Ye, Pengzhou Chen, Tao Chen. *Are Optimizing Model Hyperparameter and Software Configuration Alike? Insights from 2573 Cases*. Under submission.

## Table of Contents

- [Introduction](#introduction)
- [Repository Contents](#repository-contents)
- [Code and Quick Start](#code-and-quick-start)
- [Problems](#problems)
- [Raw Experiment Results](#raw-experiment-results)
- [RQ Supplementary](#rq-supplementary)
- [Citation](#citation)

## Introduction

Hyperparameter optimization (HPO) and software configuration tuning (SCT) are often treated as similar black-box optimization problems. However, it remains unclear whether optimizers designed for one domain should be used as state-of-the-art competitors in the other domain, how these optimizers behave under cross-domain settings, and which problem characteristics explain their effectiveness and efficiency differences.

To study these questions, we conduct a large-scale empirical study involving 31 optimizers and 83 SCT/HPO problems, resulting in 2,573 experimental cases. Under a unified experimental protocol, we compare SCT optimizers, HPO optimizers, and general-purpose optimizers in terms of effectiveness and efficiency. We further use fitness landscape analysis to explain the observed cross-domain behavior and derive practical guidelines for optimizer selection.

## Repository Contents

- `optimizer/`: implementations of the studied optimizers.
  - `SCT_optimizers/`: optimizers originally designed for SCT.
  - `HPO_optimizers/`: optimizers originally designed for HPO.
  - `general_optimizers/`: general-purpose optimizers.
  - `util/`: utility functions for querying tabular benchmark problems and saving traces.
- `experiments/`: scripts for budget preparation and main experiment execution.
  - `pre_experiment/`: pre-experiment scripts and budget computation utilities.
  - `main_experiment/`: main experiment driver.
- `analysis/`: scripts and lightweight processed tables for the research-question analysis.
  - `effectiveness/`: effectiveness ranking scripts and processed outputs.
  - `efficiency/`: efficiency ranking scripts, processed outputs, and reconstruction utilities.
  - `rq4/`: fitness landscape analysis.
  - `solution/`: optimizer recommendation analysis.
- `data/README.md`: placeholder describing where to place external benchmark problems.
- `requirements.txt`: Python dependencies used by the experiment and analysis scripts.

Large artifacts are not tracked directly in Git. The full benchmark problems and raw experiment outputs should be downloaded from the external Zenodo artifact archive and placed under the expected local directories.

## Code and Quick Start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some analysis scripts use `rpy2`, which also requires a working R installation and the corresponding R packages used by the local Scott-Knott analysis.

After preparing the external problem archive, the expected data layout is:

```text
data/
  HPO_problems/
    multi-fidelity/
    single-fidelity/
  SCT_problems/
```

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

Generated outputs are written to local result directories and are not tracked in Git.

## Problems

Our study uses 83 benchmark problems from both SCT and HPO domains, covering 36 SCT problems and 47 HPO problems. Details of the studied problems are reported in the paper.

The problems are collected or derived from the following public benchmark suites, papers, and artifact repositories:

| Domain | Source | Used for |
| --- | --- | --- |
| HPO | [HPOBench](https://github.com/automl/HPOBench) | Multi-fidelity XGBoost, neural-network, and random-forest HPO problems. |
| HPO | [Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization](https://github.com/automl/nas_benchmarks) | FCNet tabular HPO benchmarks. |
| HPO | [JAHS-Bench-201](https://github.com/automl/jahs_bench_201) | Joint architecture and hyperparameter search benchmarks. |
| HPO | [Predictable Scale: Part I](https://github.com/step-law/steplaw) | Neural-network scaling-law HPO data. |
| SCT | [Accuracy Can Lie: model-impact artifact](https://github.com/ideas-labo/model-impact) | Multiple configurable-system measurement problems reused from the TSE 2025 study. |
| SCT | [DeepPerf](https://github.com/DeepPerf/DeepPerf) | Configurable-system performance data used by the DeepPerf ICSE 2019 study. |
| SCT | [Performance Evolution of Configurable Software Systems](https://github.com/ChristianKaltenecker/PerformanceEvolution_Website) | Versioned configurable-system performance measurements. |
| SCT | [VEER / multiobj artifact](https://github.com/anonymous12138/multiobj) | Multi-objective configurable-system measurements used by the VEER study. |
| SCT | [deeparch-xplorer](https://github.com/pooyanjamshidi/deeparch-xplorer) | DeepArch configurable-system measurements. |
| SCT | [CM-CASL](https://github.com/xdbdilab/CM-CASL) | Configurable-system problems such as Spark, Redis, Hadoop, and Tomcat. |
| SCT | [Twins or False Friends? replication package](https://conf.researchr.org/details/icse-2023/icse-2023-artifact-evaluation/43/Twins-or-False-Friends-A-Study-on-Energy-Consumption-and-Performance-of-Configurable) | Energy and performance measurements for configurable software. |

## Raw Experiment Results

The benchmark problems and raw experiment traces reported in the paper can be found at:

https://doi.org/10.5281/zenodo.22096568

Recommended Zenodo archive layout:

```text
hpo-vs-sct_artifact/
  data/
    HPO_problems/
      multi-fidelity/
      single-fidelity/
    SCT_problems/
  results/
    pre/
    main/
  analysis/
    effectiveness/
    efficiency/
    rq4/
    solution/
```

The raw main experiment traces follow this naming rule:

```text
results/main/[seed]/[problem]/[optimizer]_[problem]_seed[seed].csv
```

Example:

```text
results/main/105/nginx/ATConf_nginx_seed105.csv
```

The pre-experiment traces follow this naming rule:

```text
results/pre/[seed]/[problem]/[optimizer]_[problem]_seed[seed].csv
```

Example:

```text
results/pre/1/nginx/ATConf_nginx_seed1.csv
```

The processed RQ tables used by the paper are included in `analysis/`. The full raw `data/` and `results/` directories are excluded from Git because they contain large benchmark files and more than 90,000 generated CSV traces.

## RQ Supplementary

The `analysis/` directory contains lightweight supplementary artifacts for checking the reported rankings, landscape statistics, and recommendation results:

- `analysis/effectiveness/rq1/`: transferability of SCT and HPO optimizers in terms of effectiveness.
- `analysis/effectiveness/rq2/`: comparison with general-purpose optimizers in terms of effectiveness.
- `analysis/effectiveness/rq3/`: one-for-all optimizer analysis in terms of effectiveness.
- `analysis/efficiency/rq1/`: transferability of SCT and HPO optimizers in terms of efficiency.
- `analysis/efficiency/rq2/`: comparison with general-purpose optimizers in terms of efficiency.
- `analysis/efficiency/rq3/`: one-for-all optimizer analysis in terms of efficiency.
- `analysis/rq4/landscape/`: fitness landscape analysis.
- `analysis/solution/`: optimizer recommendation analysis.

