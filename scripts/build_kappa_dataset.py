r"""Build the Kappa benchmark dataset: 103 PhononDB crystals and their PBE thermal conductivity at 300 K.

The crystals, supercells and q-point meshes are those of Matbench Discovery's κ_SRME task (B. Póta et al.,
arXiv:2408.00755; ``2024-11-09-phononDB-PBE-103-structures.extxyz``, Figshare file 52179965, CC BY 4.0).
The DFT reference is recomputed here from PhononDB's PBE force sets (A. Togo, phono3py input data of 103
compounds on MDR at NIMS, CC BY 4.0; A. Togo, L. Chaput, I. Tanaka, Phys. Rev. B 91, 094306 (2015)) with
the functions the benchmark uses for the MLIP: symmetrized force constants and the Wigner transport
equation in the relaxation-time approximation with isotope scattering. Matbench Discovery's own reference
was computed with phono3py 3.30, whose Wigner solver phono3py 4 replaced; recomputing it keeps reference
and prediction on the same solver. Its values are kept as ``published_kappa`` for comparison.

Usage (the inputs are downloaded to temp/, which git ignores)::

    python scripts/build_kappa_dataset.py temp/phonondb-pbe \\
        temp/matbench-discovery/2024-11-09-phononDB-PBE-103-structures.extxyz \\
        temp/matbench-discovery/2024-11-09-kappas-phononDB-PBE-noNAC.json.gz \\
        src/matcalc/benchmarks/data/phonondb-pbe-kappa.json.gz --workers 4 --cache temp/kappa-cache

The first directory holds ``<structure type>-<formula>.yaml.xz`` (PhononDB's ``phono3py_params.yaml.xz``)
for every compound of https://github.com/atztogo/phonondb/blob/main/mdr/phono3py_103compounds_fd_PBE/README.md.
The whole build takes about 45 minutes on a 10-core laptop (Apple M4); with ``--cache`` the result of every
compound is kept there and not recomputed by a later run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STRUCTURE_TYPES = {"rocksalt": 225, "zincblende": 216, "wurtzite": 186}
TEMPERATURE = 300.0


def reference_conductivity(job: tuple[str, str, list[int]]) -> dict:
    """PBE conductivity of one compound from its PhononDB force sets."""
    import phono3py

    from matcalc.properties.thermal_conductivity import wigner_conductivity

    path, material_id, mesh = job
    ph3 = phono3py.load(path, produce_fc=False, is_nac=False, log_level=0)
    ph3.mesh_numbers = mesh
    ph3.produce_fc2()
    ph3.symmetrize_fc2()
    ph3.produce_fc3()
    ph3.symmetrize_fc3()
    conductivity = wigner_conductivity(ph3, temperature=TEMPERATURE)
    cell = ph3.unitcell
    return {
        "material_id": material_id,
        "unit_cell": {
            "lattice": np.asarray(cell.cell).tolist(),
            "symbols": list(cell.symbols),
            "positions": np.asarray(cell.positions).tolist(),
        },
        "kappa": conductivity.kappa.tolist(),
        "mode_kappa": [[float(f"{x:.6g}") for x in row] for row in conductivity.mode_kappa],
        "weights": conductivity.weights.tolist(),
    }


def _reduced(formula: str) -> str:
    from ase.formula import Formula

    return Formula(formula).reduce()[0].format("metal")


def main() -> None:
    """Compute the reference of every compound and write the dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("phonondb")
    parser.add_argument("structures")
    parser.add_argument("published")
    parser.add_argument("target")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, help="directory keeping the result of every compound")
    args = parser.parse_args()

    from ase.io import read

    structures = read(args.structures, index=":", format="extxyz")
    with gzip.open(args.published, "rt") as f:
        published = {entry["material_id"]: entry["kappa_tot_avg"][0] for entry in json.load(f)}
    by_key = {(_reduced(a.get_chemical_formula()), int(a.info["symm.no"])): a for a in structures}
    jobs, names = [], {}
    for path in sorted(Path(args.phonondb).glob("*.yaml.xz")):
        kind, formula = path.name.removesuffix(".yaml.xz").split("-", 1)
        atoms = by_key[(_reduced(formula), STRUCTURE_TYPES[kind])]
        material_id = atoms.info["material_id"]
        names[material_id] = path.name.removesuffix(".yaml.xz")
        jobs.append((str(path), material_id, [int(n) for n in atoms.info["q_point_mesh"]]))
    if len(jobs) != len(structures):
        raise SystemExit(f"{len(jobs)} force sets for {len(structures)} structures")
    computed = {}
    if args.cache is not None:
        args.cache.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            cached = args.cache / f"{job[1]}.json"
            if cached.exists():
                computed[job[1]] = json.loads(cached.read_text())
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 1) // args.workers)))
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(reference_conductivity, [job for job in jobs if job[1] not in computed]):
            computed[result["material_id"]] = result
            if args.cache is not None:
                (args.cache / f"{result['material_id']}.json").write_text(json.dumps(result))

    entries = []
    for atoms in structures:
        info = atoms.info
        result = computed[info["material_id"]]
        # The conductivity was computed on PhononDB's unit cell: it must be the structure of the task.
        unit_cell = result["unit_cell"]
        same_lattice = np.allclose(unit_cell["lattice"], atoms.cell[:], atol=1e-6)
        if not same_lattice or list(unit_cell["symbols"]) != atoms.get_chemical_symbols():
            raise SystemExit(f"{info['material_id']}: PhononDB's unit cell differs from the task's structure")
        entries.append(
            {
                "mp_id": info["material_id"],
                "name": names[info["material_id"]],
                "formula": info["name"],
                "space_group": int(info["symm.no"]),
                "lattice": atoms.cell[:].tolist(),
                "species": atoms.get_chemical_symbols(),
                "positions": atoms.positions.tolist(),
                "fc2_supercell": np.asarray(info["fc2_supercell"]).tolist(),
                "fc3_supercell": np.asarray(info["fc3_supercell"]).tolist(),
                # The task gives it for the cubic cells; the wurtzite unit cells are primitive.
                "primitive_matrix": np.asarray(info.get("primitive_matrix", np.eye(3))).tolist(),
                "q_point_mesh": [int(n) for n in info["q_point_mesh"]],
                "kappa": float(np.mean(result["kappa"][:3])),
                "published_kappa": float(published[info["material_id"]]),
                "kappa_tensor": result["kappa"],
                "mode_kappa": result["mode_kappa"],
                "weights": result["weights"],
            }
        )
    import phono3py

    dataset = {
        "description": (
            "Kappa benchmark: 103 rock-salt, zinc-blende and wurtzite crystals (PhononDB, Matbench Discovery's "
            "kappa_SRME task) with PBE unit cells (Angstrom), supercell and primitive matrices, q-point meshes, and "
            "the PBE lattice thermal conductivity at 300 K (W/(m K)): Voigt tensor, mean of the diagonal, and the "
            "direction-averaged conductivity of every mode (irreducible q-point x band, including the q-point weight). "
            "Recomputed from "
            f"PhononDB's PBE force sets with phono3py {phono3py.__version__} (Wigner transport, RTA, isotope "
            "scattering); published_kappa is Matbench Discovery's value (phono3py 3.30)."
        ),
        "source": "https://figshare.com/articles/dataset/22715158; https://github.com/atztogo/phonondb (MDR at NIMS)",
        "license": "CC BY 4.0",
        "reference": "B. Póta et al., arXiv:2408.00755; A. Togo, L. Chaput, I. Tanaka, Phys. Rev. B 91, 094306 (2015)",
        "entries": entries,
    }
    target = Path(args.target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, separators=(",", ":"))
    print(f"{len(entries)} crystals written to {target}")


if __name__ == "__main__":
    main()
