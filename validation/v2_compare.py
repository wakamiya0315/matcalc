"""V2: compare two result tables of the same benchmark, material by material."""

import sys

import numpy as np
import pandas as pd

a, b = pd.read_csv(sys.argv[1]), pd.read_csv(sys.argv[2])
key = a.columns[0]
merged = a.merge(b, on=key, suffixes=("_a", "_b"))
print(f"{sys.argv[1]} vs {sys.argv[2]}: {len(a)} / {len(b)} rows, {len(merged)} matched on {key}")
columns = [c for c in a.columns if c.endswith("_mace") and c in b.columns and not c.startswith("status")]
for col in columns:
    va = pd.to_numeric(merged[f"{col}_a"], errors="coerce").to_numpy()
    vb = pd.to_numeric(merged[f"{col}_b"], errors="coerce").to_numpy()
    same_nan = bool(np.array_equal(np.isnan(va), np.isnan(vb)))
    ok = ~np.isnan(va) & ~np.isnan(vb)
    diff = np.abs(va[ok] - vb[ok])
    rel = diff / np.maximum(np.abs(vb[ok]), 1e-300)
    print(f"  {col:22s} n={ok.sum():3d}  max|diff|={diff.max() if diff.size else 0:.3e}  max rel={rel.max() if rel.size else 0:.3e}  same NaN pattern={same_nan}")
