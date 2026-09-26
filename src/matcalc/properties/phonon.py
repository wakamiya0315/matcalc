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
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.phonon.thermal_properties import ThermalProperties
from phonopy.units import Kb, THzToEv
from pymatgen.io.phonopy import get_phonopy_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

MESH_CHUNK_BYTES = 2**28
"""Dynamical matrices held at once while the frequencies on the q-point mesh are computed (256 MiB)."""

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
            is primitive (phonopy 4 would otherwise search for a primitive cell). Thermal properties are
            per mole of primitive cells.
        symprec: Symmetry tolerance of phonopy/spglib (Å).

    Returns:
        A ``Phonopy`` object without displacements yet.
    """
    return Phonopy(
        get_phonopy_structure(unit_cell),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
        primitive_matrix="P" if primitive_matrix is None else np.asarray(primitive_matrix, dtype=float),
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
        min_frequency: Lowest phonon frequency at the stability q-points of ``stability_qpoints`` (THz);
            negative values are imaginary modes.
    """

    heat_capacity: float
    temperature: float
    min_frequency: float

    @property
    def dynamically_stable(self) -> bool:
        """No imaginary mode below -50 K (``IMAGINARY_THRESHOLD_THZ``) at the stability q-points."""
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
        The heat capacity and the lowest frequency at the stability q-points.
    """
    phonon.forces = forces
    phonon.produce_force_constants(calculate_full_force_constants=False)
    phonon.run_qpoints(stability_qpoints(phonon))
    min_frequency = float(np.min(phonon.qpoints.frequencies))
    thermal = ThermalProperties(_mesh_frequencies(phonon, mesh))  # phonopy's sums over the mesh
    thermal.temperatures = [temperature]
    thermal.run()
    heat_capacity = float(thermal.thermal_properties[3][0])
    return HarmonicProperties(heat_capacity=heat_capacity, temperature=temperature, min_frequency=min_frequency)


def stability_qpoints(phonon: Phonopy) -> np.ndarray:
    """The q-points at which the reference judges dynamical stability, in phonopy's primitive basis.

    These are the points commensurate with the supercell matrix in units of the unit cell (for a
    diagonal matrix ``S``: all ``(n1/S1, n2/S2, n3/S3)``), used, as in the reference's calculation, as
    reduced coordinates of the primitive reciprocal lattice. For a unit cell that is not primitive this
    is not exactly the set of q-points commensurate with the supercell, but it reproduces the
    frequencies stored with the DFT reference (96 % of those compounds within 0.05 THz, against 31 % for
    the transformed points).

    Args:
        phonon: Phonopy object from ``make_phonopy``.

    Returns:
        The q-points, shape ``(n, 3)``.
    """
    return get_commensurate_points(np.rint(phonon.supercell_matrix).astype(int))


@dataclass
class _MeshFrequencies:
    """What phonopy's ``ThermalProperties`` reads from a mesh: frequencies (THz) and weights of the
    irreducible q-points, and the position of Gamma among them.
    """

    frequencies: np.ndarray
    weights: np.ndarray
    gamma_index: int | None
    dynamical_matrix: object  # only its primitive cell is read (for plots)


def _mesh_frequencies(phonon: Phonopy, mesh: Sequence[int]) -> _MeshFrequencies:
    """Frequencies on the irreducible q-points of a mesh, computed a few hundred q-points at a time.

    phonopy's own mesh calculation builds the dynamical matrices of all q-points at once, which takes
    several GB for a large low-symmetry primitive cell on a 20 x 20 x 20 mesh; the results are the same.
    """
    phonon.init_mesh(list(mesh), with_eigenvectors=False)  # q-points and weights only
    grid = phonon.mesh
    n_modes = 3 * len(phonon.primitive)
    chunk = max(1, MESH_CHUNK_BYTES // (16 * n_modes**2))  # complex dynamical matrices of 16 bytes per entry
    frequencies = []
    for start in range(0, len(grid.qpoints), chunk):
        phonon.run_qpoints(grid.qpoints[start : start + chunk])
        frequencies.append(phonon.qpoints.frequencies)
    return _MeshFrequencies(np.concatenate(frequencies), grid.weights, grid.gamma_index, phonon.dynamical_matrix)
