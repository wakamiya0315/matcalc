"""Conversions between pymatgen ``Structure`` and ASE ``Atoms``."""

from __future__ import annotations

from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def to_ase_atoms(structure: Structure | Atoms) -> Atoms:
    """Return a new ASE ``Atoms`` for ``structure``.

    The result is always a fresh object, so a calculator can be attached to it without touching
    the caller's structure.

    Args:
        structure: pymatgen ``Structure`` or ASE ``Atoms``.

    Returns:
        A new ASE ``Atoms`` with the same cell, species and positions.
    """
    return structure.copy() if isinstance(structure, Atoms) else AseAtomsAdaptor.get_atoms(structure)


def to_pmg_structure(structure: Structure | Atoms) -> Structure:
    """Return ``structure`` as a pymatgen ``Structure`` (unchanged if it already is one).

    Args:
        structure: pymatgen ``Structure`` or ASE ``Atoms``.

    Returns:
        The equivalent pymatgen ``Structure``.
    """
    return structure if isinstance(structure, Structure) else AseAtomsAdaptor.get_structure(structure)
