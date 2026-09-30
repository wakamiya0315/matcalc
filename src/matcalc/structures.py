"""Conversions between pymatgen ``Structure`` and ASE ``Atoms``, and molecules in a periodic box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

if TYPE_CHECKING:
    from collections.abc import Sequence

MOLECULE_BOX = 50.0
"""Smallest edge of the cubic periodic cell around a molecule (Å)."""


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


def molecule_in_box(
    symbols: Sequence[str], positions: Sequence[Sequence[float]] | np.ndarray, *, charge: int = 0, multiplicity: int = 1
) -> Atoms:
    """A molecule or molecular complex centred in a cubic periodic cell.

    Simulators work with periodic cells; the cell is large enough (at least ``MOLECULE_BOX`` Å and 40 Å more
    than the molecule) that periodic images lie far outside the reach of an MLIP, which then gives the
    energy of the isolated molecule. The total charge and the spin multiplicity are kept in ``info``
    (``"charge"``, ``"spin"``), where charge-aware ASE calculators read them.

    Args:
        symbols: Chemical symbols.
        positions: Cartesian positions (Å).
        charge: Total charge.
        multiplicity: Spin multiplicity 2S + 1.

    Returns:
        The molecule in its cell.
    """
    atoms = Atoms(symbols, positions=positions)
    extent = float(np.ptp(atoms.positions, axis=0).max()) if len(atoms) > 1 else 0.0
    atoms.set_cell([max(MOLECULE_BOX, extent + 40.0)] * 3)
    atoms.center()
    atoms.pbc = True
    atoms.info["charge"] = charge
    atoms.info["spin"] = multiplicity
    return atoms
