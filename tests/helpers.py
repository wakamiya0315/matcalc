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


SOFTENING_FACTOR = 1.25
"""The made-up "DFT" forces are the EMT forces times this, so the softening scale must be 1/1.25 = 0.8."""
