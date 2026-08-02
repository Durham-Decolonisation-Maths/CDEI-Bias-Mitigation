# CDEI-Bias-Mitigation

This repository contains an analysis of bias mitigation algorithms in machine learning, applied to the [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) dataset. Each notebook trains a baseline income-prediction model, applies one fairness intervention, and compares the result against the baseline on both accuracy and fairness metrics.

## Notebooks

All analysis lives in `notebooks/finance/interventions/`:

| Notebook | Stage | Technique |
| --- | --- | --- |
| `ftu.ipynb` | Pre-processing (baseline) | Fairness Through Unawareness — drop the protected attribute |
| `kamiran_calders.ipynb` | Pre-processing | Reweighing (Kamiran & Calders, 2012) |
| `hardt.ipynb` | Post-processing | Equalised-odds decision correction (Hardt et al., 2016) |

`ftu.ipynb` and `kamiran_calders.ipynb` are best taught together — FTU shows a naive pre-processing approach that only partly works, and reweighing shows a principled one on the same data. `hardt.ipynb` then shows what a post-processing fix looks like instead of a pre-processing one.

## Running the notebooks

```bash
pip install -r requirements.txt
jupyter notebook notebooks/finance/interventions/
```

This installs the small local `helpers/` package (plotting and metric utilities used by every notebook) along with `aif360`, `fairlearn`, and the usual data-science stack. Versions are intentionally unpinned — see the comment in `requirements.txt` if you need a fully frozen environment.

## Artifacts

`artifacts/` contains preprocessed Adult data (one-hot encoded, train/val/test split) and a trained baseline model, committed for reproducibility so every intervention notebook starts from the same starting point. See `artifacts/README.md` for details.
