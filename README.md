# Leakage-Resistant Phishing URL Detection: Feature Engineering, Unsupervised Analysis, and Domain-Grouped Classifier Evaluation

Code, data and paper for the WIECON-ECE 2026 paper by
**Umme Habiba Moushi, Takia Labiba Razna and Ahmed Abdal Shafi Rasel**
(Department of Computer Science and Engineering, East West University, Dhaka, Bangladesh).

The paper evaluates six classifiers on URL-Phish (115,231 deduplicated URLs) under a
protocol that groups **both** the outer train/test split **and** the inner folds of
out-of-fold target encoding by domain. Its main findings:

- Row-wise inner encoding folds cut phishing recall by 5–23 percentage points on
  unseen domains for five of six models, even under a domain-grouped outer split.
- Under grouped folds at both levels the RBF-SVM leads the engineered-feature models
  (96.92% accuracy, 94.57% phishing recall), selected by training-only grouped CV.
- Raw-URL character n-grams score higher F1, and XGBoost/HistGradientBoosting match
  the RBF-SVM — but none of the URL-Phish-trained pipelines transfers to
  [PhreshPhish](https://arxiv.org/abs/2507.10854): the SVM's ROC-AUC falls from
  0.993 to 0.762 and its false-positive rate rises from 2.3% to 41.9%, because
  URL-Phish's benign URLs are short, mostly-HTTP landing pages while realistic
  benign URLs are long HTTPS deep links.

## Repository layout

| Path | Contents |
|---|---|
| `main-first-part.ipynb` | EDA, the six classifiers, and all experiments of the paper: E0 data audit, E1 controlled split/encoding comparison (A/B/C), E2 feature ablations, E3 repeated grouped splits + Wilcoxon, E4 training-only model selection (recall-first), E5 K-Means with Cramér's V, E6 final test metrics + confusion matrices + domain-bootstrap CIs. Cells are labelled with the reviewer point they answer. |
| `baselines.ipynb` | XGBoost, HistGradientBoosting and TF-IDF character 3–5-gram SVM/LR on the same grouped splits. |
| `cross-dataset-evaluation.ipynb` | Trains the six fixed pipelines and applies them unchanged to PhreshPhish; includes the feature-extractor calibration and base-rate precision analysis. |
| `data/Dataset.csv` | The URL-Phish dataset (116,600 raw rows; 115,231 after deduplication). Also on [Mendeley Data](https://data.mendeley.com/datasets/65z9twcx3r/2). |
| `checks/` | Standalone verification scripts and their outputs (see below). |
| `paper/` | LaTeX source, figure, and compiled PDF of the paper. |

## Requirements

`pandas numpy scikit-learn matplotlib seaborn scipy category_encoders xgboost tldextract joblib`

The notebooks were written for Kaggle (dataset at `/kaggle/input/...`); to run
locally, point the data path at `data/Dataset.csv`. The saved outputs in the
notebooks are the runs reported in the paper (headline split seed 42; repeated
splits seeds 0–4; model selection by grouped 5-fold CV on the training partition
only, phishing recall first and balanced accuracy second — see `SELECT_METRIC`
in `main-first-part.ipynb`). A full run of `main-first-part.ipynb` takes about
an hour on a CPU (the RBF-SVM cells dominate).

## Checks

- `checks/tld_check.py` — verifies two dataset claims on the full CSV with
  `tldextract`: exactly two `dom` values (`cgs.gov.cz`, `szpi.gov.cz`) are not
  registrable domains, and no `dom` value (or registrable domain derived from it)
  is shared across the grouped split. Output: `checks/tld_check_result.json`.
- `checks/e4_rerun.py` — reruns the training-only model selection with the
  recall-first criterion and confirms it selects the RBF-SVM (recall 0.9477 vs
  Random Forest 0.9429 on scikit-learn 1.9.0). Output:
  `checks/e4_rerun_result.json`.

## Citation

If you use this code or dataset description, please cite:

> U. H. Moushi, T. L. Razna, and A. A. S. Rasel, "Leakage-Resistant Phishing URL
> Detection: Feature Engineering, Unsupervised Analysis, and Domain-Grouped
> Classifier Evaluation," in *2026 IEEE International Women in Engineering (WIE)
> Conference on Electrical and Computer Engineering (WIECON-ECE)*, Dhaka,
> Bangladesh, 2026.

```bibtex
@inproceedings{moushi2026leakage,
  author    = {Moushi, Umme Habiba and Razna, Takia Labiba and Rasel, Ahmed Abdal Shafi},
  title     = {Leakage-Resistant Phishing URL Detection: Feature Engineering, Unsupervised Analysis, and Domain-Grouped Classifier Evaluation},
  booktitle = {2026 IEEE International Women in Engineering (WIE) Conference on Electrical and Computer Engineering (WIECON-ECE)},
  address   = {Dhaka, Bangladesh},
  year      = {2026}
}
```

External resources: URL-Phish on
[Mendeley Data](https://data.mendeley.com/datasets/65z9twcx3r/2) (DOI
10.17632/65z9twcx3r.2); PhreshPhish ([arXiv:2507.10854](https://arxiv.org/abs/2507.10854));
[PhishTank](https://phishtank.org); [ROR data](https://doi.org/10.5281/zenodo.6347574) on Zenodo.
