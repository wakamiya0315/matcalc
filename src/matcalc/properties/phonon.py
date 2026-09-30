"""Harmonic phonons by finite displacements (phonopy): heat capacity and dynamical stability.

The calculation follows the one behind the DFT reference of the Phonon benchmark: phonopy with a given
unit cell, supercell matrix and primitive matrix. Each displaced atom of the displacement list gives one
supercell; the forces on all atoms of all displaced supercells give the force constants, and the phonon
frequencies on a q-point mesh give the heat capacity in the harmonic approximation.

With the crystal's symmetry (the default), phonopy displaces only symmetry-inequivalent atoms along few
directions, completes the displacement-force pairs with the site symmetry of the displaced atom and copies
the force constants to equivalent atoms with the space group, rotations, inversion and mirrors alike. This
is exact for forces that transform with every operation of the space group, as DFT forces and those of
O(3)-equivariant MLIPs do. It is not for MLIPs whose forces change under inversion or mirrors
(SO(3)-equivariant or non-equivariant models), in two ways:

- the forces left on the relaxed cell do not follow its symmetry (a relaxation under a symmetry constraint
  stops on the symmetric part of the forces), and phonopy, which takes the force of a displaced supercell
  for the response to the displacement, turns them into force constants of order (residual force) /
  (amplitude). Subtracting the forces on the undisplaced supercell (``undisplaced_supercell``) removes this
  part; for an MLIP whose forces follow the symmetry it changes nothing, since phonopy's symmetrization
  already cancels a symmetric residual;
- the displacement-force pairs that phonopy completes and copies with inversion and mirrors are not the
  model's own. Without symmetry (``use_symmetry=False`` of ``make_phonopy``) phonopy still uses the lattice
  translations, which hold for any MLIP, and every atom of the primitive cell is displaced by plus and minus
  the amplitude along the three lattice directions (central differences, in which the residual forces
  cancel too).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.phonon.thermal_properties import ThermalProperties
from phonopy.physical_units import get_physical_units
from pymatgen.io.phonopy import get_phonopy_structure

if TYPE_CHECKING:
    from collections.abc import Sequence

    from phonopy.harmonic.displacement import Type1DisplacementDataset
    from phonopy.phonon.mesh import Mesh
    from pymatgen.core import Structure

MESH_CHUNK_BYTES = 2**28
"""Dynamical matrices held at once while the frequencies on the q-point mesh are computed (256 MiB)."""

IMAGINARY_THRESHOLD_THZ = 50.0 * get_physical_units().KB / get_physical_units().THzToEv
"""Frequencies below minus this (the frequency of 50 K, 1.04 THz) count as imaginary modes, as in the
stability criterion of the DFT reference (A. Loew et al., npj Comput. Mater. 2025)."""


def make_phonopy(
    unit_cell: Structure,
    supercell_matrix: Sequence[Sequence[int]],
    primitive_matrix: Sequence[Sequence[float]] | None = None,
    *,
    symprec: float = 1e-5,
    use_symmetry: bool = True,
) -> Phonopy:
    """Set up phonopy for a unit cell.

    Args:
        unit_cell: The (relaxed) unit cell.
        supercell_matrix: Supercell lattice vectors in units of the unit cell's (phonopy's convention).
        primitive_matrix: Primitive lattice vectors in units of the unit cell's; ``None`` if the unit cell
            is primitive (phonopy 4 would otherwise search for a primitive cell). Thermal properties are
            per mole of primitive cells.
        symprec: Symmetry tolerance of phonopy/spglib (Å).
        use_symmetry: Let phonopy use the point operations of the space group (see the module
            docstring); with ``False`` it uses the lattice translations only.

    Returns:
        A ``Phonopy`` object without displacements yet.
    """
    return Phonopy(
        get_phonopy_structure(unit_cell),
        supercell_matrix=np.asarray(supercell_matrix, dtype=int),
        primitive_matrix="P" if primitive_matrix is None else np.asarray(primitive_matrix, dtype=float),
        symprec=symprec,
        is_symmetry=use_symmetry,
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
    dataset: Type1DisplacementDataset = {
        "natom": len(phonon.supercell),
        "first_atoms": [
            {"number": int(row[0]), "displacement": np.asarray(row[1:4], dtype=float)} for row in displacements
        ],
    }
    phonon.dataset = dataset
    supercell = phonon.supercell
    cells = []
    for row in displacements:
        positions = supercell.positions.copy()
        positions[int(row[0])] += np.asarray(row[1:4], dtype=float)
        cells.append(Atoms(numbers=supercell.numbers, cell=supercell.cell, positions=positions, pbc=True))
    return cells


def undisplaced_supercell(phonon: Phonopy) -> Atoms:
    """The supercell without displacement, with the atoms in the order of the displaced ones.

    Args:
        phonon: Phonopy object from ``make_phonopy``.

    Returns:
        The supercell as ASE ``Atoms``; its forces are those left on the relaxed cell.
    """
    supercell = phonon.supercell
    return Atoms(numbers=supercell.numbers, cell=supercell.cell, positions=supercell.positions.copy(), pbc=True)


def generated_displacements(
    phonon: Phonopy, distance: float, *, plus_minus: bool | Literal["auto"] = "auto"
) -> list[list[float]]:
    """Displacements generated by phonopy for the symmetry it finds, as rows like those of the datasets.

    Args:
        phonon: Phonopy object from ``make_phonopy``.
        distance: Displacement amplitude (Å).
        plus_minus: Also displace by minus the amplitude: ``"auto"`` (phonopy's default) only where the
            symmetry does not already give the opposite displacement, ``True`` always.

    Returns:
        One row per displaced supercell: index of the displaced atom in phonopy's supercell (from 0), then
        its displacement vector (Å).
    """
    phonon.generate_displacements(distance=distance, is_plusminus=plus_minus)
    dataset = phonon.dataset
    if dataset is None or "first_atoms" not in dataset:
        raise RuntimeError("phonopy generated no displacements")
    first_atoms = cast("Type1DisplacementDataset", dataset)["first_atoms"]  # one displaced atom per supercell
    return [[d["number"], *d["displacement"]] for d in first_atoms]


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
    min_frequency = float(np.min(_frequencies_at(phonon, stability_qpoints(phonon))))
    # phonopy's sums over the mesh, fed with the frequencies computed chunk by chunk
    thermal = ThermalProperties(cast("Mesh", _mesh_frequencies(phonon, mesh)))
    thermal.temperatures = [temperature]
    thermal.run()
    if thermal.thermal_properties is None:
        raise RuntimeError("phonopy computed no thermal properties")
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
    if grid is None:
        raise RuntimeError("phonopy set up no q-point mesh")
    n_modes = 3 * len(phonon.primitive)
    chunk = max(1, MESH_CHUNK_BYTES // (16 * n_modes**2))  # complex dynamical matrices of 16 bytes per entry
    frequencies = [
        _frequencies_at(phonon, grid.qpoints[start : start + chunk]) for start in range(0, len(grid.qpoints), chunk)
    ]
    return _MeshFrequencies(np.concatenate(frequencies), grid.weights, grid.gamma_index, phonon.dynamical_matrix)


def _frequencies_at(phonon: Phonopy, qpoints: np.ndarray) -> np.ndarray:
    """Phonon frequencies (THz) at some q-points, shape (number of q-points, number of modes)."""
    phonon.run_qpoints(qpoints)
    if phonon.qpoints is None:
        raise RuntimeError("phonopy computed no frequencies")
    return phonon.qpoints.frequencies
