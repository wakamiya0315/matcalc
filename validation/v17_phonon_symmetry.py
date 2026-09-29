"""V17: the Phonon benchmark with and without the crystal's symmetry, for one MLIP.

Runs ``PhononBenchmark`` on the same compounds in three settings and writes the three tables:

- ``before``: the protocol before 2026-09-28 (the DFT's displacements with phonopy's symmetry, the forces
  taken as they are; ``subtract_residual_forces=False``);
- ``default``: the same minus the forces on the undisplaced supercell;
- ``no-symmetry``: ``use_symmetry=False`` (plus and minus displacements of every atom of the primitive cell,
  no symmetry; exact for any MLIP), minus the same forces.

The MLIP is not part of matcalc: ``loader`` is ``module:function``, a function that takes ``model_path``
and returns a TorchSim model (or an ASE calculator). The compounds are ``--n-random`` drawn with ``--seed``
plus those listed in ``--ids`` (one mp_id per line).

Usage::

    PYTHONPATH=<fork>/src python v17_phonon_symmetry.py run <loader> <model_path> <name> <out_dir> \\
        [--n-random 30] [--ids ids.txt] [--seed 42] [--workers 3]
    python v17_phonon_symmetry.py compare <out_dir> <name>
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import random
import time
from pathlib import Path

import pandas as pd

SETTINGS = {
    "before": {"subtract_residual_forces": False},
    "default": {},
    "no-symmetry": {"use_symmetry": False},
}


def subset_dataset(out_dir: Path, n_random: int, ids_file: str | None, seed: int) -> Path:
    """A local copy of the packaged dataset with the chosen compounds only."""
    from matcalc.benchmarks.phonon import DATASET

    with gzip.open(DATASET, "rt") as f:
        raw = json.load(f)
    entries = {e["mp_id"]: e for e in raw["entries"]}
    chosen = random.Random(seed).sample(sorted(entries), n_random) if n_random else []
    if ids_file:
        chosen += [m for m in Path(ids_file).read_text().split() if m not in chosen]
    path = out_dir / "compounds.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump(raw | {"entries": [entries[m] for m in chosen]}, f)
    return path


def run(args: argparse.Namespace) -> None:
    from matcalc import PhononBenchmark
    from matcalc.simulation import as_simulator

    module, function = args.loader.split(":")
    simulator = as_simulator(getattr(importlib.import_module(module), function)(args.model_path))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = subset_dataset(args.out_dir, args.n_random, args.ids, args.seed)
    for label, settings in SETTINGS.items():
        benchmark = PhononBenchmark(dataset, workers=args.workers, **settings)
        began = time.perf_counter()
        table = benchmark.run(simulator, args.name)
        table.to_csv(args.out_dir / f"{args.name}.{label}.csv", index=False)
        summary = benchmark.summarize(table, args.name)
        print(json.dumps({"setting": label, "seconds": round(time.perf_counter() - began, 1), "summary": summary}))


def compare(out_dir: Path, name: str) -> None:
    tables = {label: pd.read_csv(out_dir / f"{name}.{label}.csv") for label in SETTINGS}
    exact = tables["no-symmetry"]
    for label, table in tables.items():
        stable_dft = table["stable_DFT"].astype(bool)
        stable = table[f"stable_{name}"].astype(bool)
        d_cv = (table[f"CV_{name}"] - exact[f"CV_{name}"]).abs()
        d_min = (table[f"min_frequency_{name}"] - exact[f"min_frequency_{name}"]).abs()
        print(
            f"{label:12s} compounds {len(table)}  FU {int((stable_dft & ~stable).sum())}  "
            f"FS {int((~stable_dft & stable).sum())}  MAE C_V {(table[f'CV_{name}'] - table['CV_DFT']).abs().mean():.2f}  "
            f"vs no-symmetry: |dC_V| median {d_cv.median():.3g} max {d_cv.max():.3g}, "
            f"|d min_frequency| max {d_min.max():.3g} THz"
        )
    if f"residual_force_{name}" in exact:
        print("largest residual force (eV/A): median", round(exact[f"residual_force_{name}"].median(), 4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run the three settings")
    runner.add_argument("loader", help="module:function returning a TorchSim model or an ASE calculator")
    runner.add_argument("model_path")
    runner.add_argument("name")
    runner.add_argument("out_dir", type=Path)
    runner.add_argument("--n-random", type=int, default=30)
    runner.add_argument("--ids", help="file with more mp_ids, one per line")
    runner.add_argument("--seed", type=int, default=42)
    runner.add_argument("--workers", type=int, default=3)
    comparer = commands.add_parser("compare", help="compare the tables of a run")
    comparer.add_argument("out_dir", type=Path)
    comparer.add_argument("name")
    args = parser.parse_args()
    if args.command == "compare":
        compare(args.out_dir, args.name)
    else:
        run(args)


if __name__ == "__main__":
    main()
