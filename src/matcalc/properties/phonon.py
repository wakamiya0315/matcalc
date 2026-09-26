"""Harmonic phonons by finite displacements (phonopy): heat capacity and dynamical stability.

The calculation follows the one behind the DFT reference of the Phonon benchmark: phonopy with a given
unit cell, supercell matrix and primitive matrix. Each displaced atom of the displacement list gives one
supercell; the forces on all atoms of all displaced supercells give the force constants, and the phonon
frequencies on a q-point mesh give the heat capacity in the harmonic approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import DynmatToForceConstants
from phonopy.units import Kb, THzToEv
from pymatgen.io.phonopy import get_phonopy_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

IMAGINARY_THRESHOLD_THZ = 50.0 * Kb / THzToEv
"""Frequencies below minus this (the frequency of 50 K, 1.04 THz) count as imaginary modes, as in the
stability criterion of the DFT reference (A. Loew et al., npj Comput. Mater. 2025)."""


def make_phonopy(
    unit_cell: Structure,
    supercell_matrix: Sequence[Sequence[int]],
    primitive_matrix: Sequence[Sequence[float]] | None = None,
    *,
    symprec: float = 1e-5,
) -> Phonopy:
    """Set up phonopy for a unit cell.

    Args:
        unit_cell: The (relaxed) unit cell.
        supercell_matrix: Supercell lattice vectors in units of the unit cell's (phonopy's convention).
        primitive_matrix: Primitive lattice vectors in units of the unit cell's; ``None`` if the unit cell
            is primitive. Thermal properties are per mole of primitive cells.
        symprec: Symmetry tolerance of phonopy/spglib (Å).

    Returns:
        A ``Phonopy`` object without displacements yet.
    """
    return Phonopy(
        get_phonopy_structure(unit_cell),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
        primitive_matrix=None if primitive_matrix is None else np.asarray(primitive_matrix, dtype=float),
        symprec=symprec,
    )


def displaced_supercells(phonon: Phonopy, displacements: Sequence[Sequence[float]]) -> list[Atoms]:
    """The displaced supercells of a list of displacements.

    Args:
        phonon: Phonopy object from ``make_phonopy``.
        displacements: One row per displaced supercell: index of the displaced atom in phonopy's supercell
            (from 0), then its displacement vector (Å).

    Returns:
        One ASE ``Atoms`` per displacement, in the order given.
    """
    phonon.dataset = {
        "natom": len(phonon.supercell),
        "first_atoms": [{"number": int(row[0]), "displacement": [float(x) for x in row[1:4]]} for row in displacements],
    }
    supercell = phonon.supercell
    cells = []
    for row in displacements:
        positions = supercell.positions.copy()
        positions[int(row[0])] += np.asarray(row[1:4], dtype=float)
        cells.append(Atoms(numbers=supercell.numbers, cell=supercell.cell, positions=positions, pbc=True))
    return cells


@dataclass
class HarmonicProperties:
    """Harmonic phonon properties of one compound.

    Attributes:
        heat_capacity: Constant-volume heat capacity C_V at ``temperature`` (J/(K·mol) per mole of
            primitive cells).
        temperature: Temperature (K).
        min_frequency: Lowest phonon frequency at the q-points commensurate with the supercell (THz);
            negative values are imaginary modes.
    """

    heat_capacity: float
    temperature: float
    min_frequency: float

    @property
    def dynamically_stable(self) -> bool:
        """No imaginary mode below -50 K (``IMAGINARY_THRESHOLD_THZ``) at the commensurate q-points."""
        return self.min_frequency >= -IMAGINARY_THRESHOLD_THZ


def harmonic_properties(
    phonon: Phonopy,
    forces: Sequence[np.ndarray],
    *,
    temperature: float = 300.0,
    mesh: Sequence[int] = (20, 20, 20),
) -> HarmonicProperties:
    """Force constants → frequencies → heat capacity and dynamical stability.

    Only the compact force constants (rows of the atoms in the primitive cell) are built, and no
    eigenvectors are kept: both give the same frequencies as the full arrays with far less memory.

    Args:
        phonon: Phonopy object whose displacements were set by ``displaced_supercells``.
        forces: Forces on every atom of every displaced supercell (eV/Å), same order.
        temperature: Temperature of the heat capacity (K).
        mesh: q-point mesh for the heat capacity (the reference used 20 x 20 x 20).

    Returns:
        The heat capacity and the lowest frequency at the q-points commensurate with the supercell.
    """
    phonon.forces = forces
    phonon.produce_force_constants(calculate_full_force_constants=False)
    commensurate = DynmatToForceConstants(phonon.primitive, phonon.supercell).commensurate_points
    phonon.run_qpoints(commensurate)
    min_frequency = float(np.min(phonon.get_qpoints_dict()["frequencies"]))
    phonon.run_mesh(list(mesh), with_eigenvectors=False)
    phonon.run_thermal_properties(temperatures=[temperature])
    heat_capacity = float(phonon.get_thermal_properties_dict()["heat_capacity"][0])
    return HarmonicProperties(heat_capacity=heat_capacity, temperature=temperature, min_frequency=min_frequency)
