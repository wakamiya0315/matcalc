"""Small crystals that ASE's EMT potential describes, for building test datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from pymatgen.core import Structure


def structure(formula: str) -> Structure:
    """Small crystals that EMT describes: fcc Cu (4 atoms), B2 NiAl, L1_2 Cu3Au, L1_0 CuAu, primitive Cu."""
    if formula == "Cu":
        atoms = bulk("Cu", "fcc", a=3.62, cubic=True)
    elif formula == "Cu1":
        atoms = bulk("Cu", "fcc", a=3.62)
    elif formula == "NiAl":
        atoms = bulk("NiAl", "cesiumchloride", a=2.88)
    elif formula == "Cu3Au":
        atoms = bulk("Cu", "fcc", a=3.75, cubic=True)
        atoms.symbols[0] = "Au"
    elif formula == "CuAu":
        atoms = bulk("Cu", "fcc", a=3.85, cubic=True)
        atoms.symbols[[0, 1]] = "Au"
    else:
        raise ValueError(formula)
    return AseAtomsAdaptor.get_structure(atoms)


FCC_PRIMITIVE = [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
"""phonopy's primitive matrix of a face-centred cubic unit cell."""


def phonon_entry(
    material_id: str,
    formula: str,
    supercell_matrix: list[list[int]],
    primitive_matrix: list[list[float]] | None,
    heat_capacity: float,
    *,
    stable: bool,
    strain: float = 0.0,
) -> dict:
    """An entry of the Phonon dataset format, with the displacements phonopy generates (0.01 A).

    ``strain`` stretches the unit cell isotropically (relative change of the lattice constants).
    """
    from phonopy import Phonopy
    from pymatgen.io.phonopy import get_phonopy_structure

    unit_cell = structure(formula).copy()
    unit_cell.scale_lattice(unit_cell.volume * (1 + strain) ** 3)
    phonon = Phonopy(
        get_phonopy_structure(unit_cell), supercell_matrix=supercell_matrix, primitive_matrix=primitive_matrix
    )
    phonon.generate_displacements(distance=0.01)
    return {
        "mp_id": material_id,
        "formula": formula,
        "lattice": unit_cell.lattice.matrix.tolist(),
        "species": [site.specie.symbol for site in unit_cell],
        "frac_coords": unit_cell.frac_coords.tolist(),
        "supercell_matrix": supercell_matrix,
        "primitive_matrix": primitive_matrix,
        "displacements": [[d["number"], *d["displacement"]] for d in phonon.dataset["first_atoms"]],
        "symprec": 1e-5,
        "space_group": phonon.symmetry.dataset.number,
        "heat_capacity": heat_capacity,
        "stable": stable,
    }


SOFTENING_FACTOR = 1.25
"""The made-up "DFT" forces are the EMT forces times this, so the softening scale must be 1/1.25 = 0.8."""
