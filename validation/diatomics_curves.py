"""Diatomic curves of a MACE test model in Matbench Discovery's prediction format, compared point by point with
published predictions of the same model.

Usage: python diatomics_curves.py {fork-ase,fork-torchsim} OUT.json.gz [--model medium] [--published FILE]
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("code", choices=["fork-ase", "fork-torchsim"])
    parser.add_argument("out")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--published", help="published predictions, e.g. mace-mp-0-2026-06-28-diatomics.json.gz")
    args = parser.parse_args()

    import matcalc
    from matcalc.properties.diatomics import DIMER_DISTANCES, WALL_LOWER, dimers, evaluation_window
    from matcalc.simulation.torchsim import TorchSimSimulator

    sys.path.insert(0, str(Path(__file__).parent))
    from mace_models import load_mace

    backend = "torchsim" if args.code == "fork-torchsim" else "ase"
    model = load_mace(backend, model=args.model)
    simulator = TorchSimSimulator(model, show_progress=False) if backend == "torchsim" else matcalc.ASESimulator(model, show_progress=False)
    elements = [m.material_id for m in matcalc.DiatomicsBenchmark().materials]
    start = time.perf_counter()
    results = simulator.single_point([d for el in elements for d in dimers(el)], compute_stress=False)
    elapsed = time.perf_counter() - start
    n = len(DIMER_DISTANCES)
    curves = {}
    for k, el in enumerate(elements):
        own = results[k * n : (k + 1) * n]
        curves[f"{el}-{el}"] = {
            "energies": [r.energy if r.error is None else None for r in own],
            "forces": [r.forces.tolist() if r.error is None and r.forces is not None else None for r in own],
        }
    with gzip.open(args.out, "wt") as f:
        json.dump({"distances": DIMER_DISTANCES.tolist(), "homo-nuclear": curves, "hetero-nuclear": {}}, f)
    print(f"{len(elements)} curves ({len(results)} single points) in {elapsed:.1f} s")
    if not args.published:
        return
    with gzip.open(args.published, "rt") as f:
        published = json.load(f)["homo-nuclear"]
    rows = []
    for formula, curve in curves.items():
        if formula not in published or not published[formula]["energies"]:
            continue
        el = formula.split("-")[0]
        wall_min, r_max = evaluation_window(el, float(DIMER_DISTANCES.max()), WALL_LOWER)
        scored = (DIMER_DISTANCES >= wall_min - 1e-12) & (DIMER_DISTANCES <= r_max)
        e = np.array([np.nan if v is None else v for v in curve["energies"]], dtype=float)
        e_pub = np.asarray(published[formula]["energies"], dtype=float)
        f_ours = np.array([np.full((2, 3), np.nan) if v is None else v for v in curve["forces"]], dtype=float)
        f_pub = np.asarray(published[formula]["forces"], dtype=float)
        rows.append((el, np.nanmax(np.abs(e - e_pub)[scored]), np.nanmax(np.abs(f_ours - f_pub)[scored]), np.nanmax(np.abs(e - e_pub))))
    de, df, de_all = (np.array([r[i] for r in rows]) for i in (1, 2, 3))
    print(f"{len(rows)} curves compared; in the scored range: max |dE| median {np.median(de):.2e} eV, largest {de.max():.2e} ({rows[int(np.argmax(de))][0]}); "
          f"max |dF| median {np.median(df):.2e} eV/A, largest {df.max():.2e} ({rows[int(np.argmax(df))][0]}); whole grid: largest |dE| {de_all.max():.2e} ({rows[int(np.argmax(de_all))][0]})")


if __name__ == "__main__":
    main()
