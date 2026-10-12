"""Verify the paper's tldextract claims on the full dataset (E0 audit was skipped on Kaggle).

Checks:
1. dom values that are NOT a registrable domain (the paper claims exactly two)
2. registrable-domain overlap across the grouped train/test split (paper claims none)
3. dom-level overlap (paper claims none)
"""
import json
import sys

import numpy as np
import pandas as pd
import tldextract
from sklearn.model_selection import GroupShuffleSplit

csv = sys.argv[1]
ext = tldextract.TLDExtract(suffix_list_urls=())

df = pd.read_csv(csv)
out = {"raw_rows": len(df)}

for tag, d in [("raw", df), ("deduped", df.drop_duplicates().reset_index(drop=True))]:
    reg = d["dom"].map(lambda h: ext(str(h)).registered_domain or str(h))
    mism = d.loc[reg != d["dom"], ["dom"]]
    out[f"{tag}_rows"] = len(d)
    out[f"{tag}_not_registrable_count"] = int(len(mism))
    out[f"{tag}_not_registrable_values"] = sorted(set(mism["dom"].tolist()))

d = df.drop_duplicates().reset_index(drop=True)
y = d["label"].astype(int)
X = d[["dom"]].copy()
tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, groups=X["dom"]))
dom_tr, dom_te = set(d.iloc[tr]["dom"]), set(d.iloc[te]["dom"])
out["dom_overlap_grouped_split"] = len(dom_tr & dom_te)

reg_all = d["dom"].map(lambda h: ext(str(h)).registered_domain or str(h))
out["registrable_overlap_grouped_split"] = len(set(reg_all.iloc[tr]) & set(reg_all.iloc[te]))
out["unique_dom"] = int(d["dom"].nunique())

print(json.dumps(out, indent=1))
with open(sys.argv[2], "w") as f:
    json.dump(out, f, indent=2)
