"""V4/V5: per-material agreement of two result tables against the tolerances of the plan."""

import json
import sys

import numpy as np
import pandas as pd

TOLERANCES = {"K_vrh_mace": 1.0, "G_vrh_mace": 1.0, "CV_mace": 0.5, "Eform_mace": 0.005, "d_mace": 0.01,
              "softening_scale_mace": 0.01}
reference, candidate = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
key = reference.columns[0]
merged = reference.merge(candidate, on=key, suffixes=("_ref", "_new"))
report = {"n_ref": len(reference), "n_new": len(candidate), "n_matched": len(merged), "quantities": {}}
for col, tol in TOLERANCES.items():
    if f"{col}_ref" not in merged:
        continue
    a = pd.to_numeric(merged[f"{col}_ref"], errors="coerce").to_numpy()
    b = pd.to_numeric(merged[f"{col}_new"], errors="coerce").to_numpy()
    nan_mismatch = merged.loc[np.isnan(a) != np.isnan(b), key].tolist()
    both = ~np.isnan(a) & ~np.isnan(b)
    diff = np.abs(a - b)
    outliers = merged.loc[both & (diff > tol), key].tolist()
    report["quantities"][col] = {
        "tolerance": tol, "n_compared": int(both.sum()),
        "within": float(np.mean(diff[both] <= tol)) if both.any() else None,
        "max|diff|": float(diff[both].max()) if both.any() else None,
        "median|diff|": float(np.median(diff[both])) if both.any() else None,
        "nan_mismatch": nan_mismatch, "outliers": {k: float(d) for k, d in zip(outliers, diff[both & (diff > tol)])},
    }
print(json.dumps(report, indent=1))
