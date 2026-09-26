"""Build the Phonon benchmark dataset from the Alexandria PBE phonon calculations.

The benchmark reuses the settings of the DFT reference (A. Loew et al., npj Comput. Mater. (2025),
doi:10.1038/s41524-025-01650-1; data: https://alexandria.icams.rub.de/data/phonon_benchmark/, CC BY 4.0):
the unit cell relaxed with PBE, the supercell matrix, the primitive matrix and the displacements of
phonopy. This script extracts them, with the DFT heat capacity at 300 K and the DFT stability flag, for
the 1,170 binary compounds of upstream matcalc's Phonon benchmark.

Usage (the inputs are downloaded to temp/, which git ignores)::

    python scripts/build_phonon_dataset.py temp/alexandria src/matcalc/benchmarks/data/alexandria-pbe-phonon.json.gz

The input directory must contain ``pbe/<mp_id>.yaml.bz2`` (phonopy files of Alexandria), ``pbe.csv``
(Alexandria's summary table, for the stability flag) and ``formulas.json`` (material id to formula of
upstream's dataset ``alexandria-binary-pbe-phonon-2025.1.json.gz``, which defines the 1,170 compounds).
Needs PyYAML.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import json
import re
import sys
from pathlib import Path

import yaml

TEMPERATURE = 300.0
LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _sections(text: str) -> dict[str, str]:
    """Split a phonopy YAML file into its top-level sections (without parsing the large ones)."""
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^([a-z_]+):", text, re.MULTILINE)]
    starts.append((len(text), ""))
    return {name: text[a:b] for (a, name), (b, _) in zip(starts, starts[1:], strict=False)}


def _load(section: str) -> object:
    return next(iter(yaml.safe_load(section).values()))


def read_compound(path: Path) -> dict:
    """The unit cell, supercell and primitive matrices, displacements and C_V(300 K) of one compound."""
    parts = _sections(bz2.open(path, "rt").read())
    unit_cell = _load(parts["unit_cell"])
    temperatures = _load(parts["temperatures"])
    # Only the displaced atom and its displacement are needed, not the DFT forces.
    displacements = [
        [int(atom) - 1, float(dx), float(dy), float(dz)]  # phonopy files number atoms from 1
        for atom, dx, dy, dz in re.findall(
            r"- atom: (\d+)\n  displacement:\n  - (\S+)\n  - (\S+)\n  - (\S+)", parts["displacements"]
        )
    ]
    phonopy_info = _load(parts["phonopy"])
    # Frequencies (THz) at the q-points where the reference judged dynamical stability
    min_frequency = min(min(row) for row in yaml.load(parts["phonon_freq"], Loader=LOADER)["phonon_freq"])
    return {
        "lattice": unit_cell["lattice"],
        "species": [point["symbol"] for point in unit_cell["points"]],
        "frac_coords": [point["coordinates"] for point in unit_cell["points"]],
        "supercell_matrix": _load(parts["supercell_matrix"]),
        "primitive_matrix": _load(parts["primitive_matrix"]) if "primitive_matrix" in parts else None,
        "displacements": displacements,
        "heat_capacity": _load(parts["heat_capacity"])[temperatures.index(TEMPERATURE)],
        "min_frequency": float(min_frequency),
        "space_group": _load(parts["space_group"])["number"],
        "symprec": phonopy_info["symmetry_tolerance"],
        "phonopy_version": phonopy_info["version"],
    }


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    formulas = json.loads((source / "formulas.json").read_text())
    with (source / "pbe.csv").open() as f:
        stable = {row["mp_id"]: row["stable"] == "True" for row in csv.DictReader(f)}
    entries = []
    for mp_id, formula in formulas.items():
        entry = {"mp_id": mp_id, "formula": formula} | read_compound(source / "pbe" / f"{mp_id}.yaml.bz2")
        entry["stable"] = stable[mp_id]
        entries.append(entry)
    dataset = {
        "description": (
            "Phonon benchmark: 1,170 binary compounds with the settings and results of the Alexandria PBE "
            "phonon calculations (phonopy): unit cell, supercell matrix, primitive matrix, displacements "
            "(0-based supercell atom index and vector in Angstrom), heat capacity at 300 K in J/(K mol) "
            "per mole of primitive cells, lowest frequency (THz) at the q-points of the stability test, and "
            "the dynamical stability flag of Alexandria."
        ),
        "source": "https://alexandria.icams.rub.de/data/phonon_benchmark/pbe/",
        "license": "CC BY 4.0",
        "reference": "A. Loew, D. Sun, H.-C. Wang, S. Botti, M. A. L. Marques, npj Comput. Mater. (2025), "
        "doi:10.1038/s41524-025-01650-1",
        "entries": entries,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as f:
        json.dump(dataset, f, separators=(",", ":"))
    print(f"{len(entries)} compounds written to {target}")


if __name__ == "__main__":
    main()
