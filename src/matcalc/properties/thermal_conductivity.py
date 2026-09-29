"""Lattice thermal conductivity from harmonic and third-order force constants (phono3py).

The calculation is the one behind Matbench Discovery's κ_SRME task (B. Póta, P. Ahlawat, G. Csányi,
M. Simoncelli, arXiv:2408.00755; code: github.com/MPA2suite/k_SRME and matbench-discovery): phono3py with
finite displacements of 0.01 Å, symmetrized force constants, and the Wigner transport equation of
Simoncelli, Marzari and Mauri (2019) in the relaxation-time approximation (particle-like plus coherence
conductivity) with isotope scattering, at 300 K on the q-point mesh of the DFT reference. Matbench
Discovery pins phono3py 3.30, whose "MS-SMM19" transport was replaced by "SMM19" in phono3py 4 (which
the rest of this package needs); the per-mode conductivities are rebuilt here the same way.
Needs phono3py >= 4.7.

The error of a predicted conductivity is measured mode by mode: the symmetric relative mean error
SRME = 2 Σ_modes |κ_pred,mode - κ_DFT,mode| / Σ_q w_q / (κ_pred + κ_DFT), between 0 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import Atoms

if TYPE_CHECKING:
    from collections.abc import Sequence

    from phono3py import Phono3py

KAPPA_ERROR_MAX = 2.0
"""Largest symmetric relative error; given to failed or censored predictions."""

IMAGINARY_FREQUENCY_THRESHOLD = -0.01
"""Acoustic modes at Γ below this frequency (THz) count as imaginary (numerical noise lies above it)."""


def make_phono3py(
    unit_cell: Atoms,
    fc2_supercell: Sequence[Sequence[int]],
    fc3_supercell: Sequence[Sequence[int]],
    mesh: Sequence[int],
    *,
    displacement: float = 0.01,
    symprec: float = 1e-5,
) -> Phono3py:
    """Set up phono3py and its displacements for a unit cell.

    Args:
        unit_cell: The (relaxed) unit cell.
        fc2_supercell: Supercell matrix of the harmonic force constants (in units of the unit cell).
        fc3_supercell: Supercell matrix of the third-order force constants.
        mesh: q-point mesh of the conductivity.
        displacement: Displacement amplitude (Å).
        symprec: Symmetry tolerance of phono3py (Å).

    Returns:
        A ``Phono3py`` object with its displacements generated (plus and minus where phono3py decides).
    """
    from phono3py import Phono3py
    from phonopy.structure.atoms import PhonopyAtoms

    phono3py = Phono3py(
        PhonopyAtoms(symbols=unit_cell.get_chemical_symbols(), cell=unit_cell.cell[:], positions=unit_cell.positions),
        supercell_matrix=np.asarray(fc3_supercell, dtype=int),
        phonon_supercell_matrix=np.asarray(fc2_supercell, dtype=int),
        primitive_matrix="auto",
        symprec=symprec,
    )
    phono3py.mesh_numbers = list(mesh)
    phono3py.generate_displacements(distance=displacement, is_plusminus="auto")
    return phono3py


def fc2_supercells(phono3py: Phono3py) -> list[Atoms]:
    """The displaced supercells of the harmonic force constants, in phono3py's order."""
    return [_atoms(cell) for cell in phono3py.phonon_supercells_with_displacements]


def fc3_supercells(phono3py: Phono3py) -> list[Atoms | None]:
    """The displaced supercells of the third-order force constants (``None`` for pairs phono3py skips)."""
    return [None if cell is None else _atoms(cell) for cell in phono3py.supercells_with_displacements]


def harmonic_frequencies(phono3py: Phono3py, fc2_forces: Sequence[np.ndarray]) -> np.ndarray:
    """Harmonic force constants (symmetrized) and the frequencies on the whole q-point grid.

    Args:
        phono3py: Object from ``make_phono3py``.
        fc2_forces: Forces on the atoms of every supercell of ``fc2_supercells`` (eV/Å), same order.

    Returns:
        Frequencies (THz), shape (grid points, bands); the first grid point is Γ.
    """
    phono3py.phonon_forces = np.asarray(fc2_forces)
    phono3py.produce_fc2(symmetrize_fc2=True)
    phono3py.init_phph_interaction(symmetrize_fc3q=False)
    phono3py.run_phonon_solver()
    return np.asarray(phono3py.get_phonon_data()[0])


def has_imaginary_modes(frequencies: np.ndarray) -> bool:
    """Imaginary modes anywhere on the grid, tolerating numerical noise of the acoustic modes at Γ."""
    return bool(
        np.all(np.isnan(frequencies))
        or np.any(frequencies[0, 3:] < 0)
        or np.any(frequencies[0, :3] < IMAGINARY_FREQUENCY_THRESHOLD)
        or np.any(frequencies[1:] < 0)
    )


@dataclass
class Conductivity:
    """Lattice thermal conductivity at one temperature.

    Attributes:
        kappa: Total conductivity tensor in Voigt order xx, yy, zz, yz, xz, xy (W/(m K)).
        kappa_particle: Its particle-like (Peierls-Boltzmann, intra-band) part.
        kappa_coherence: Its wave-like (coherence, inter-band) part.
        mode_kappa: Direction-averaged conductivity of every mode (irreducible q-point x band),
            including the q-point weight (W/(m K)): the particle-like part of the mode plus its share
            of the coherence with every other band, in proportion to the two heat capacities.
        weights: Weights of the irreducible q-points.
    """

    kappa: np.ndarray
    kappa_particle: np.ndarray
    kappa_coherence: np.ndarray
    mode_kappa: np.ndarray
    weights: np.ndarray

    @property
    def kappa_average(self) -> float:
        """Mean of the diagonal components xx, yy, zz (W/(m K))."""
        return float(np.mean(self.kappa[:3]))


def thermal_conductivity(
    phono3py: Phono3py, fc3_forces: Sequence[np.ndarray | None], *, temperature: float = 300.0
) -> Conductivity:
    """Third-order force constants and the Wigner-RTA conductivity (after ``harmonic_frequencies``).

    Args:
        phono3py: Object from ``make_phono3py`` whose harmonic force constants are set.
        fc3_forces: Forces on every supercell of ``fc3_supercells`` (zeros where it is ``None``).
        temperature: Temperature (K).

    Returns:
        The conductivity.

    Raises:
        RuntimeError: If phono3py returns no conductivity.
    """
    n_atoms = len(phono3py.supercell)
    phono3py.forces = np.asarray([np.zeros((n_atoms, 3)) if f is None else f for f in fc3_forces])
    phono3py.produce_fc3(symmetrize_fc3r=True)
    return wigner_conductivity(phono3py, temperature=temperature)


def wigner_conductivity(phono3py: Phono3py, *, temperature: float = 300.0) -> Conductivity:
    """The Wigner-RTA conductivity of a ``Phono3py`` object whose second- and third-order force constants are set.

    Used both for the MLIP (``thermal_conductivity``) and for the DFT reference (from PhononDB's force sets).

    Args:
        phono3py: The ``Phono3py`` object.
        temperature: Temperature (K).

    Returns:
        The conductivity.

    Raises:
        RuntimeError: If phono3py returns no conductivity.
    """
    phono3py.init_phph_interaction(symmetrize_fc3q=False)
    phono3py.run_thermal_conductivity(
        temperatures=[temperature], is_isotope=True, transport_type="SMM19", boundary_mfp=1e6
    )
    kappa: Any = phono3py.thermal_conductivity
    if kappa is None or kappa.kappa is None or kappa.mode_heat_capacities is None:
        raise RuntimeError("phono3py computed no thermal conductivity")
    # (q-point, band, band, Voigt) at the only sigma and temperature: diagonal = particle-like part
    matrix = np.asarray(kappa.mode_kappa_matrix)[0, 0]
    heat_capacity = np.asarray(kappa.mode_heat_capacities)[0]
    return Conductivity(
        kappa=np.asarray(kappa.kappa)[0, 0],
        kappa_particle=np.asarray(kappa.kappa_intra)[0, 0],
        kappa_coherence=np.asarray(kappa.kappa_inter)[0, 0],
        mode_kappa=_mode_kappa(matrix, heat_capacity)[..., :3].mean(axis=-1),
        weights=np.asarray(kappa.grid_weights),
    )


def _mode_kappa(matrix: np.ndarray, heat_capacity: np.ndarray) -> np.ndarray:
    """Conductivity of every mode from the band-pair matrix of the Wigner transport equation.

    The coherence between bands i and j (both off-diagonal elements) is split between the two modes in
    proportion to their heat capacities, as Matbench Discovery does.

    Args:
        matrix: Band-pair conductivities, shape (q-points, bands, bands, 6).
        heat_capacity: Mode heat capacities, shape (q-points, bands).

    Returns:
        Mode conductivities, shape (q-points, bands, 6).
    """
    bands = np.arange(matrix.shape[1])
    diagonal = matrix[:, bands, bands, :]
    coherence = matrix.copy()
    coherence[:, bands, bands, :] = 0.0
    pairs = coherence + coherence.transpose(0, 2, 1, 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = heat_capacity[:, :, None] / (heat_capacity[:, :, None] + heat_capacity[:, None, :])
    return diagonal + (pairs * np.nan_to_num(share, nan=0.0)[..., None]).sum(axis=2)


def symmetric_relative_difference(predicted: float, reference: float) -> float:
    """SRD = 2 (κ_pred - κ_DFT) / (κ_pred + κ_DFT), between -2 and 2."""
    total = predicted + reference
    return 0.0 if total == 0 else KAPPA_ERROR_MAX * (predicted - reference) / total


def symmetric_relative_mean_error(
    predicted: Conductivity, reference_mode_kappa: np.ndarray, reference_kappa: float
) -> float:
    """Mode-resolved error SRME = 2 Σ|κ_pred,mode - κ_DFT,mode| / Σ w / (κ_pred + κ_DFT).

    Args:
        predicted: Predicted conductivity.
        reference_mode_kappa: DFT conductivity of every mode (same irreducible q-points and bands).
        reference_kappa: DFT conductivity (mean of the diagonal, W/(m K)).

    Returns:
        SRME between 0 and 2; ``KAPPA_ERROR_MAX`` if the modes do not correspond.
    """
    reference_mode_kappa = np.asarray(reference_mode_kappa)
    if predicted.mode_kappa.shape != reference_mode_kappa.shape or predicted.weights.sum() <= 0:
        return KAPPA_ERROR_MAX
    microscopic_error = np.abs(predicted.mode_kappa - reference_mode_kappa).sum() / predicted.weights.sum()
    total = predicted.kappa_average + reference_kappa
    if total == 0:
        return 0.0 if microscopic_error == 0 else KAPPA_ERROR_MAX
    return min(KAPPA_ERROR_MAX, KAPPA_ERROR_MAX * microscopic_error / total)


def _atoms(cell: Any) -> Atoms:
    return Atoms(cell.symbols, cell=cell.cell, positions=cell.positions, pbc=True)
