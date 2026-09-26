"""Check the Phonon pipeline with the DFT forces: does it give back the DFT heat capacities and stability?

For every compound of the packaged dataset, the DFT forces on the displaced supercells are read from
Alexandria's phonopy file and passed through the benchmark's own phonopy step (DFT unit cell, supercell
and primitive matrices, displacements, 20x20x20 mesh, -50 K stability test).

Usage: python phonon_dft_check.py <directory with pbe/<mp_id>.yaml.bz2> <workers> <output.json>
"""

import bz2
import json
import re
import sys
from multiprocessing import Pool

import numpy as np
import yaml

from matcalc.benchmarks.phonon import PhononBenchmark, PhononJob, harmonic_properties_of

LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def dft_forces(path: str) -> list[np.ndarray]:
    text = bz2.open(path, "rt").read()
    start = text.index("displacements:")
    end = re.search(r"^[a-z_]+:", text[start + 15 :], re.MULTILINE).start() + start + 15
    entries = yaml.load(text[start:end], Loader=LOADER)["displacements"]
    return [np.array(entry["forces"], dtype=float) for entry in entries]


def check(args: tuple) -> dict:
    directory, material = args
    forces = dft_forces(f"{directory}/pbe/{material.material_id}.yaml.bz2")
    job = PhononJob(material.structure, forces, **material.settings, temperature=300.0, mesh=(20, 20, 20))
    harmonic = harmonic_properties_of(job)
    return {
        "mp_id": material.material_id,
        "CV": harmonic.heat_capacity,
        "CV_DFT": material.reference["CV"],
        "stable": harmonic.dynamically_stable,
        "stable_DFT": material.reference["stable"],
        "min_frequency": harmonic.min_frequency,
    }


def main() -> None:
    directory, workers, output = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    materials = PhononBenchmark().materials
    with Pool(workers) as pool:
        rows = pool.map(check, [(directory, m) for m in materials], chunksize=4)
    json.dump(rows, open(output, "w"))
    d = np.array([abs(r["CV"] - r["CV_DFT"]) for r in rows])
    print(f"{len(rows)} compounds: |CV - CV_DFT| median {np.median(d):.2e}, p99 {np.percentile(d, 99):.2e}, max {d.max():.2e}")
    flips = [r for r in rows if r["stable"] != r["stable_DFT"]]
    print(f"stability flag differs for {len(flips)}:", [(r["mp_id"], round(r["min_frequency"], 3), r["stable_DFT"]) for r in flips[:10]])


if __name__ == "__main__":
    main()
