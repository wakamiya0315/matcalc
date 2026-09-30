"""GMTKN55Benchmark against the PBEh-3c results that the GMTKN55 repository publishes.

  python gmtkn55_check.py [ARCHIVE]

ARCHIVE is the repository archive (default: the one GMTKN55Benchmark downloads). The PBEh-3c energies of the
archive's ORCA outputs are given to the benchmark through a simulator that returns them, so the benchmark's
reading of the .res files (coefficients, braces, BH76RC), its reaction energies, filters and WTMAD-2 are
compared with the evaluator's published tables (_results/PBEh-3c_*.csv; their values are rounded to 0.01
kcal/mol). WTMAD-2 is also recomputed from the published reaction values, which must give the published
numbers exactly.
"""

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from ase import units

HARTREE_TO_KCAL = 627.509541
"""The factor of the GMTKN55 evaluator (tmer2)."""


class PublishedEnergies:
    """A simulator that returns the PBEh-3c energies of the molecules (in eV, from the evaluator's kcal/mol)."""

    batched = True

    def __init__(self, energies: dict[int, float]) -> None:
        self.energies = energies

    def relax(self, structures, **kwargs):  # noqa: ANN001, ANN003, ANN201, D102
        raise NotImplementedError

    def single_point(self, structures, *, compute_stress=True):  # noqa: ANN001, ANN201, ARG002, D102
        from matcalc.simulation import SinglePointResult

        return [SinglePointResult(energy=self.energies[id(s)], forces=None, stress=None) for s in structures]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", nargs="?")
    args = parser.parse_args()

    from matcalc import GMTKN55Benchmark
    from matcalc.benchmarks.gmtkn55 import GMTKN55_ARCHIVE, wtmad2
    from matcalc.datasets import download_file

    archive_path = Path(args.archive) if args.archive else download_file(GMTKN55_ARCHIVE, "gmtkn55")
    bench = GMTKN55Benchmark(archive_path)
    kcal = units.kcal / units.mol
    with zipfile.ZipFile(archive_path) as archive:
        root = archive.namelist()[0].split("/")[0]
        energies = {}
        for key, atoms in bench.molecules.items():
            out = archive.read(f"{root}/{key}/PBEh-3c/orca.out").decode(errors="replace")
            hartree = float(re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", out)[-1])
            energies[id(atoms)] = hartree * HARTREE_TO_KCAL * kcal
        published = pd.read_csv(io.BytesIO(archive.read(f"{root}/_results/PBEh-3c_reactions.csv")))
        published_wtmad2 = pd.read_csv(io.BytesIO(archive.read(f"{root}/_results/PBEh-3c_wtmad2.csv")), index_col=0)
    table = bench.run(PublishedEnergies(energies), "pbeh3c")
    published["reaction"] = [f"{s}:{k + 1}" for s, k in zip(published["Subset"], published.groupby("Subset").cumcount())]
    mine = table.set_index("reaction")
    theirs = published.set_index("reaction")
    common = mine.index.intersection(theirs.index)
    summary = bench.summarize(table, "pbeh3c")
    subsets = {
        subset: {"N": len(rows), "mean_abs_reference": float(np.mean(np.abs(rows["ReferenceValue"])))}
        | {"MAE": float(np.mean(np.abs(rows["MethodValue"] - rows["ReferenceValue"])))}
        for subset, rows in published.groupby("Subset")
    }
    total, per_category, mean_abs = wtmad2(subsets)
    names = {"small reactions": "smallreactions", "large reactions": "largereactions", "barrier heights": "barrierheights",
             "intermolecular NCI": "intermolecular", "intramolecular NCI": "intramolecular", "all NCI": "all_nci"}
    report = {
        "reactions": [len(table), len(published), len(common)],
        "max_abs_reference_difference": float((mine.loc[common, "energy_ref"] - theirs.loc[common, "ReferenceValue"]).abs().max()),
        "max_abs_energy_difference": float((mine.loc[common, "energy_pbeh3c"] - theirs.loc[common, "MethodValue"]).abs().max()),
        "WTMAD-2 (ours, unrounded)": summary["WTMAD-2"],
        "mean_abs_reference (ours, unrounded)": summary["mean_abs_reference"],
        "WTMAD-2 from the published values minus the published WTMAD-2": {
            "total": total - float(published_wtmad2.loc["total"].iloc[0]),
            **{k: per_category[k] - float(published_wtmad2.loc[v].iloc[0]) for k, v in names.items()},
        },
        "mean_abs_reference (published values)": mean_abs,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
