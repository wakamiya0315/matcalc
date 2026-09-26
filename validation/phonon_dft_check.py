"""Check the Phonon pipeline with the DFT forces: does it give back the DFT heat capacities and stability?

For every compound of the packaged dataset, the DFT forces on the displaced supercells are read from
Alexandria's phonopy file and passed through the benchmark's own phonopy step (DFT unit cell, supercell
and primitive matrices, displacements, 20x20x20 mesh, -50 K stability test).

Usage: python phonon_dft_check.py <directory with pbe/<mp_id>.yaml.bz2> <workers> <output.json>
"""

from __future__ import annotations

import bz2
import json
import re
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from matcalc.benchmarks.phonon import PhononBenchmark, PhononJob, harmonic_properties_of


def dft_forces(path: str) -> list[np.ndarray]:
    """The DFT forces of every displacement (a regular expression is much faster than a YAML parser)."""
    text = bz2.open(path, "rt").read()
    start = text.index("displacements:")
    end = re.search(r"^[a-z_]+:", text[start + 15 :], re.MULTILINE).start() + start + 15
    forces = []
    for entry in text[start:end].split("\n- atom:")[1:]:
        block = entry.split("  forces:\n", 1)[1]
        values = re.findall(r"- (-?[0-9][0-9.]*(?:[eE][-+]?[0-9]+)?)\s*$", block, re.MULTILINE)
        forces.append(np.array(values, dtype=float).reshape(-1, 3))
    return forces


def check(directory: str, material: object) -> dict:
    forces = dft_forces(f"{directory}/pbe/{material.material_id}.yaml.bz2")
    job = PhononJob(material.structure, forces, **material.settings, temperature=300.0, mesh=(20, 20, 20))
    harmonic = harmonic_properties_of(job)
    row = {
        "mp_id": material.material_id,
        "CV_DFT": material.reference["CV"],
        "stable_DFT": material.reference["stable"],
    }
    if isinstance(harmonic, str):
        return row | {"error": harmonic}
    return row | {
        "CV": harmonic.heat_capacity,
        "stable": harmonic.dynamically_stable,
        "min_frequency": harmonic.min_frequency,
    }


def main() -> None:
    directory, workers, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    materials = PhononBenchmark().materials
    rows, start = [], time.perf_counter()
    # A pool of spawned processes reports a crashed worker instead of waiting for it forever.
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(check, directory, m) for m in materials]
        for n, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if n % 100 == 0:
                print(f"{n} compounds after {time.perf_counter() - start:.0f} s", flush=True)
    json.dump(rows, open(output, "w"))
    ok = [r for r in rows if "error" not in r]
    d = np.array([abs(r["CV"] - r["CV_DFT"]) for r in ok])
    print(
        f"{len(ok)} of {len(rows)} compounds: |CV - CV_DFT| median {np.median(d):.2e}, p99 {np.percentile(d, 99):.2e}, max {d.max():.2e}"
    )
    print("errors:", [(r["mp_id"], r["error"]) for r in rows if "error" in r][:10])
    flips = [r for r in ok if r["stable"] != r["stable_DFT"]]
    print(
        f"stability flag differs for {len(flips)}:",
        [(r["mp_id"], round(r["min_frequency"], 3), r["stable_DFT"]) for r in flips[:10]],
    )


if __name__ == "__main__":
    main()
