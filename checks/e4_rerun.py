"""Rerun experiment E4 of main-first-part.ipynb with SELECT_METRIC = 'phish_rec'
(recall-first, as described in the paper's Evaluation Framework section) and report
which model the training-only grouped 5-fold CV selects.

Code mirrors the notebook verbatim (same seeds, same grouped folds, same configs).
"""
import json
import sys

import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
csv = sys.argv[1]

df = pd.read_csv(csv)
df["tld"] = df["tld"].fillna(df["tld"].mode()[0])
df = df.drop_duplicates().reset_index(drop=True)
y = df["label"].astype(int)
CAT_COLS = ["dom", "tld"]
ALL_NUM = [c for c in df.columns if c not in ("url", "dom", "tld", "label")]
Xx = df[ALL_NUM + CAT_COLS].copy()
Xx["grp"] = df["dom"]
print("rows:", len(df), flush=True)


class NestedTargetEncoder(BaseEstimator, TransformerMixin):
    """Verbatim from the notebook (grouped=True path)."""
    def __init__(self, cat_cols, group_col="grp", n_splits=5, smoothing=10, grouped=True, seed=SEED):
        self.cat_cols, self.group_col = cat_cols, group_col
        self.n_splits, self.smoothing, self.grouped, self.seed = n_splits, smoothing, grouped, seed

    def _enc(self):
        return ce.TargetEncoder(cols=self.cat_cols, smoothing=self.smoothing)

    def fit(self, X, y):
        self.enc_ = self._enc().fit(X[self.cat_cols], y)
        return self

    def fit_transform(self, X, y):
        X = X.reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        out = np.zeros((len(X), len(self.cat_cols)))
        splits = GroupKFold(self.n_splits).split(X, y, groups=X[self.group_col]) if self.grouped else None
        for a_tr, a_va in splits:
            enc = self._enc().fit(X.loc[a_tr, self.cat_cols], y.loc[a_tr])
            out[a_va] = enc.transform(X.loc[a_va, self.cat_cols]).to_numpy()
        self.fit(X, y)
        return out

    def transform(self, X):
        return self.enc_.transform(X[self.cat_cols]).to_numpy()


def make_model(name, seed=SEED):
    if name == "LR":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "DT":
        return DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=seed)
    if name == "NB":
        return GaussianNB()
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=20,
                                      min_samples_leaf=10, class_weight="balanced",
                                      random_state=seed, n_jobs=-1)
    if name == "Stacking":
        return StackingClassifier(
            estimators=[("lr", LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
                        ("dt", DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                      class_weight="balanced", random_state=seed)),
                        ("rf", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10,
                                                      class_weight="balanced", random_state=seed, n_jobs=-1))],
            final_estimator=LogisticRegression(), n_jobs=-1)
    if name == "SVM":
        return SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=seed)
    raise ValueError(name)


def build_pipe(model_name, num_cols, cat_cols, grouped_te, seed=SEED):
    tf = []
    if cat_cols:
        tf.append(("te", NestedTargetEncoder(cat_cols, grouped=grouped_te, seed=seed), cat_cols + ["grp"]))
    tf.append(("sc", StandardScaler(), num_cols))
    return Pipeline([("prep", ColumnTransformer(tf)), ("model", make_model(model_name, seed))])


def metrics(yt, pred):
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return dict(phish_rec=recall_score(yt, pred), phish_prec=precision_score(yt, pred, zero_division=0),
                phish_f1=f1_score(yt, pred), bal_acc=balanced_accuracy_score(yt, pred),
                mcc=matthews_corrcoef(yt, pred))


def fit_eval(model, tr, te, num_cols, cat_cols, grouped_te, Xd, yd, seed=SEED):
    pipe = build_pipe(model, num_cols, cat_cols, grouped_te, seed)
    pipe.fit(Xd.iloc[tr], yd.iloc[tr])
    pred = pipe.predict(Xd.iloc[te])
    return metrics(yd.iloc[te], pred)


tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(Xx, y, groups=Xx["grp"]))
Xtr, ytr = Xx.iloc[tr].reset_index(drop=True), y.iloc[tr].reset_index(drop=True)
print("train rows:", len(Xtr), flush=True)

MODELS = ["LR", "DT", "NB", "RF", "Stacking", "SVM"]
SELECT_METRIC = "phish_rec"  # recall-first, as stated in the paper (balanced accuracy secondary)
rows = []
for m in MODELS:
    f = []
    for a_tr, a_va in GroupKFold(5).split(Xtr, ytr, groups=Xtr["grp"]):
        f.append(fit_eval(m, a_tr, a_va, ALL_NUM, CAT_COLS, True, Xtr, ytr))
        print("  %s fold done: phish_rec=%.4f" % (m, f[-1]["phish_rec"]), flush=True)
    rows.append({"model": m, **{k: float(np.mean([x[k] for x in f])) for k in
                                ("phish_rec", "phish_prec", "phish_f1", "bal_acc", "mcc")}})
E4 = pd.DataFrame(rows).sort_values(SELECT_METRIC, ascending=False).reset_index(drop=True)
print(E4.round(4).to_string(), flush=True)
result = {"selected_by_recall_first": E4.loc[0, "model"],
          "table": E4.to_dict(orient="records")}
print("SELECTED (recall-first, train-only grouped CV):", E4.loc[0, "model"], flush=True)
with open(sys.argv[2], "w") as fh:
    json.dump(result, fh, indent=2)
