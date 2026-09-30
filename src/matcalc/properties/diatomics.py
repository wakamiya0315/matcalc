"""Potential-energy curves of homonuclear dimers X2 and their metrics (Matbench Discovery's diatomics task).

A curve is the energy and the forces of a dimer in a cubic 50 Å box at separations from 0.1 to 6 Å (119
log-spaced points). The metrics are those of Matbench Discovery (J. Riebesell et al., Nat. Mach. Intell. 7,
836 (2025); github.com/janosh/matbench-discovery, ``matbench_discovery/metrics/diatomics``), which build on
the smoothness analysis of the MACE-MP paper and of MLIP Arena:

- Smoothness of the predicted curve, between 0.9 r_cov and min(3.1 r_vdW, 6 Å) (covalent radius r_cov,
  van der Waals radius r_vdW of Alvarez): tortuosity (total variation of the energy over the variation
  along the shortest path through its minimum), the numbers of sign changes of the energy differences
  and of the force, the size of the steps at those changes, and the total variation of the force.
- Against the PBE curves (same window): mean absolute errors of the energy (both curves shifted to zero
  at their largest separation) and of the forces on 200 common points; errors of the equilibrium
  distance, the well depth and the harmonic vibrational wavenumber (quadratic fit of the five points
  around the minimum; for wells deeper than 0.05 eV); and the error of the repulsive wall, the
  separation at 1, 5, 10, 20, 50 and 100 eV above the minimum, down to 0.8 r_cov.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers, covalent_radii, vdw_alvarez

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

DIMER_DISTANCES = np.geomspace(0.1, 6.0, 119)
"""Separations of the curves (Å), as in Matbench Discovery."""

BOX_SIZE = 50.0
"""Edge of the cubic cell around a dimer (Å): periodic images are far outside any MLIP cutoff."""

WINDOW_LOWER = 0.9
"""Lower end of the scored window of the smoothness and error metrics, in covalent radii."""

WALL_LOWER = 0.8
"""Lower end of the repulsive-wall metric, in covalent radii (the shortest DFT separation)."""

WALL_THRESHOLDS = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)
"""Energies above the minimum (eV) at which the wall radii are compared."""

MIN_WELL_DEPTH = 0.05
"""Wells shallower than this (eV) in the reference have no bond-length, depth or frequency error."""

N_INTERPOLATION = 200
"""Points of the common grid on which energy and force errors are taken."""

MIN_WINDOW_POINTS = 5
"""A curve with fewer separations in the window gets no metrics."""

MIN_FIT_POINTS = 3
"""Points needed for the quadratic fit of the well."""

MIN_CURVE_POINTS = 2
"""Points needed to compare a curve with the reference."""

ROUGH_JUMP, ROUGH_CHANGES = 1.5, 3
"""A reference curve with energy steps of at least ROUGH_JUMP eV at ROUGH_CHANGES or more sign changes in the
window is too rough to score against."""

METRICS = (
    "pbe_energy_mae",
    "pbe_force_mae",
    "pbe_wall_dist_mae",
    "pbe_bond_length_error",
    "pbe_well_depth_error",
    "pbe_vib_freq_error",
    "tortuosity",
    "energy_diff_flips",
    "energy_jump",
    "force_flips",
    "force_total_variation",
    "force_jump",
)
"""Names of the metrics, as in Matbench Discovery; errors in eV, eV/Å, Å and cm⁻¹."""


def dimers(element: str, distances: Sequence[float] | np.ndarray = DIMER_DISTANCES) -> list[Atoms]:
    """The dimer of an element at every separation, one atom at the centre of the box, the other along x.

    Args:
        element: Chemical symbol.
        distances: Separations (Å), smaller than half the box.

    Returns:
        One structure per separation.
    """
    centre = BOX_SIZE / 2
    return [
        Atoms([element, element], positions=[[centre] * 3, [centre + d, centre, centre]], cell=[BOX_SIZE] * 3, pbc=True)
        for d in distances
    ]


def evaluation_window(element: str, largest_distance: float, lower: float = WINDOW_LOWER) -> tuple[float, float]:
    """The scored separations of an element: from ``lower`` covalent radii to min(3.1 r_vdW, largest distance).

    Args:
        element: Chemical symbol.
        largest_distance: Largest separation of the curve (Å).
        lower: Lower end in covalent radii.

    Returns:
        (r_min, r_max) in Å.
    """
    z = atomic_numbers[element]
    r_min = lower * float(covalent_radii[z])
    r_vdw = float(vdw_alvarez.vdw_radii[z])
    return r_min, min(3.1 * r_vdw, largest_distance) if np.isfinite(r_vdw) else largest_distance


def _nonzero_steps(values: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Differences of consecutive values (those below ``threshold`` dropped) and where their sign changes."""
    steps = np.diff(values)
    steps = steps[(np.abs(steps) >= threshold) & (steps != 0)]
    return steps, np.diff(np.sign(steps)) != 0


def tortuosity(energies: np.ndarray) -> float:
    """Total variation of the energy over |E(first) - E_min| + |E(last) - E_min| (1 for a single well)."""
    shortest = abs(energies[0] - energies.min()) + abs(energies[-1] - energies.min())
    return float(np.abs(np.diff(energies)).sum() / shortest) if shortest else float("nan")


def sign_changes(values: np.ndarray, threshold: float) -> float:
    """Number of sign changes of the differences of consecutive values larger than ``threshold``."""
    return float(_nonzero_steps(values, threshold)[1].sum())


def jump_size(values: np.ndarray, threshold: float) -> float:
    """Sum of the absolute differences on both sides of every sign change (those of ``sign_changes``)."""
    steps, changes = _nonzero_steps(values, threshold)
    return float(np.abs(steps[:-1][changes]).sum() + np.abs(steps[1:][changes]).sum())


def force_sign_changes(force_x: np.ndarray, threshold: float = 1e-2) -> float:
    """Number of changes of direction of the force on the first atom (forces below ``threshold`` ignored)."""
    signs = np.sign(force_x[np.abs(force_x) >= threshold])
    return float((np.diff(signs[signs != 0]) != 0).sum())


def _common_grid(
    ref_distances: np.ndarray, ref_values: np.ndarray, distances: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Both curves (energies or forces) interpolated linearly onto ``N_INTERPOLATION`` points of their overlap."""
    low, high = max(ref_distances.min(), distances.min()), min(ref_distances.max(), distances.max())
    if low >= high:
        raise ValueError("the curves do not overlap")
    grid = np.linspace(low, high, N_INTERPOLATION)

    def onto(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        columns = y.reshape(len(x), -1)
        return np.stack([np.interp(grid, x, column) for column in columns.T], axis=1).reshape(len(grid), *y.shape[1:])

    return onto(ref_distances, ref_values), onto(distances, values)


def energy_mae(
    ref_distances: np.ndarray, ref_energies: np.ndarray, distances: np.ndarray, energies: np.ndarray
) -> float:
    """Mean absolute energy error on the common grid, both curves shifted to zero at its largest separation."""
    ref, pred = _common_grid(ref_distances, ref_energies, distances, energies)
    return float(np.mean(np.abs((pred - pred[-1]) - (ref - ref[-1]))))


def force_mae(ref_distances: np.ndarray, ref_forces: np.ndarray, distances: np.ndarray, forces: np.ndarray) -> float:
    """Mean absolute error of all force components on the common grid (eV/Å)."""
    ref, pred = _common_grid(ref_distances, ref_forces, distances, forces)
    return float(np.mean(np.abs(ref - pred)))


def well_depth(energies: np.ndarray) -> float:
    """Depth of the well below the energy at the largest separation (eV)."""
    return float(energies[-1] - energies.min())


def quadratic_well(distances: np.ndarray, energies: np.ndarray, n_points: int = 5) -> tuple[float, float]:
    """Equilibrium separation (Å) and curvature (eV/Å²) of a quadratic fit of the points around the minimum.

    The separation is the lowest sampled point when the fit has no minimum inside the fitted points, and the
    curvature is NaN when the fit is not convex.
    """
    lowest = int(np.argmin(energies))
    if len(distances) < MIN_FIT_POINTS:
        return float(distances[lowest]), float("nan")
    start = min(max(0, lowest - n_points // 2), max(0, len(distances) - n_points))
    x, y = distances[start : start + n_points], energies[start : start + n_points]
    if len(x) < MIN_FIT_POINTS:
        return float(distances[lowest]), float("nan")
    a, b, _ = np.polyfit(x, y, 2)
    if a <= 0:
        return float(distances[lowest]), float("nan")
    r_eq = -b / (2 * a)
    return (float(r_eq) if x.min() <= r_eq <= x.max() else float(distances[lowest])), float(2 * a)


def vibrational_wavenumber(element: str, curvature: float) -> float:
    """Harmonic wavenumber (cm⁻¹) of a homonuclear dimer with this curvature of the energy (eV/Å²)."""
    if not np.isfinite(curvature) or curvature <= 0:
        return float("nan")
    reduced_mass = atomic_masses[atomic_numbers[element]] * 1.66053906660e-27 / 2  # kg
    angular_frequency = np.sqrt(curvature * 16.02176634 / reduced_mass)  # eV/Å² -> N/m
    return float(angular_frequency / (2 * np.pi * 2.99792458e10))


def wall_radius(distances: np.ndarray, energies: np.ndarray, height: float) -> float:
    """Separation on the repulsive side where the energy first reaches ``height`` above the minimum (Å)."""
    lowest = int(np.argmin(energies))
    if lowest == 0:
        return float("nan")
    inward = distances[lowest::-1]
    rise = np.maximum.accumulate(energies[lowest::-1] - energies[lowest])
    levels, first = np.unique(rise, return_index=True)
    if len(levels) < 2 or height > levels[-1]:  # noqa: PLR2004 - two points to interpolate between
        return float("nan")
    return float(np.interp(height, levels, inward[first]))


def wall_error(
    ref_distances: np.ndarray, ref_energies: np.ndarray, distances: np.ndarray, energies: np.ndarray
) -> float:
    """Mean error of the wall radii at the heights the reference reaches.

    Where the prediction never reaches a height, the error is the reference radius.
    """
    errors = []
    for height in WALL_THRESHOLDS:
        ref = wall_radius(ref_distances, ref_energies, height)
        if np.isfinite(ref):
            pred = wall_radius(distances, energies, height)
            errors.append(abs(pred - ref) if np.isfinite(pred) else ref)
    return float(np.mean(errors)) if errors else float("nan")


def is_rough(element: str, distances: ArrayLike, energies: ArrayLike) -> bool:
    """Whether a reference curve is too jumpy to score against.

    It is when, in the window, the energy steps at sign changes add up to at least ``ROUGH_JUMP`` eV with at least
    ``ROUGH_CHANGES`` changes, or when it has non-finite energies there.
    """
    distances, energies = np.asarray(distances, dtype=float), np.asarray(energies, dtype=float)
    r_min, r_max = evaluation_window(element, float(distances.max()))
    window = (distances >= r_min) & (distances <= r_max)
    if window.sum() < MIN_WINDOW_POINTS:
        return False
    if not np.isfinite(energies[window]).all():
        return True
    return jump_size(energies[window], 1e-3) >= ROUGH_JUMP and sign_changes(energies[window], 1e-3) >= ROUGH_CHANGES


def curve_metrics(
    element: str,
    distances: ArrayLike,
    energies: ArrayLike,
    forces: ArrayLike,
    reference: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
) -> dict[str, float]:
    """The metrics of one predicted curve.

    Args:
        element: Chemical symbol.
        distances: Separations (Å), ascending.
        energies: Predicted energies (eV).
        forces: Predicted forces, shape (separations, 2, 3) (eV/Å).
        reference: The reference curve (separations, energies, forces), or ``None`` for the smoothness metrics
            only (also when the reference is too rough, see ``is_rough``).

    Returns:
        Metric name → value (see ``METRICS``); empty when the window holds fewer than five separations or
        non-finite predictions (Matbench Discovery then leaves the element out).
    """
    distances, energies = np.asarray(distances, dtype=float), np.asarray(energies, dtype=float)
    forces = np.asarray(forces, dtype=float)
    r_min, r_max = evaluation_window(element, float(distances.max()))
    window = (distances >= r_min) & (distances <= r_max)
    finite = np.isfinite(energies[window]).all() and np.isfinite(forces[window]).all()
    if window.sum() < MIN_WINDOW_POINTS or not finite:
        return {}
    x, e, force_x = distances[window], energies[window], forces[window][:, 0, 0]
    metrics = {
        "tortuosity": tortuosity(e),
        "energy_diff_flips": sign_changes(e, 1e-3),
        "energy_jump": jump_size(e, 1e-3),
        "force_flips": force_sign_changes(force_x),
        "force_total_variation": float(np.abs(np.diff(force_x)).sum()),
        "force_jump": jump_size(force_x, 0.0),
    }
    if reference is None:
        return metrics
    ref_distances, ref_energies, ref_forces = (np.asarray(values, dtype=float) for values in reference)
    ref_window = (ref_distances >= r_min) & (ref_distances <= r_max)
    ref_x, ref_e = ref_distances[ref_window], ref_energies[ref_window]
    overlap = len(ref_x) >= MIN_CURVE_POINTS and max(ref_x.min(), x.min()) < min(ref_x.max(), x.max())
    if len(ref_x) >= MIN_CURVE_POINTS:
        if overlap:
            metrics["pbe_energy_mae"] = energy_mae(ref_x, ref_e, x, e)
        deep_enough = well_depth(ref_e) >= MIN_WELL_DEPTH
        (ref_r_eq, ref_curvature), (r_eq, curvature) = quadratic_well(ref_x, ref_e), quadratic_well(x, e)
        nan = float("nan")
        metrics["pbe_bond_length_error"] = abs(r_eq - ref_r_eq) if deep_enough else nan
        metrics["pbe_well_depth_error"] = abs(well_depth(e) - well_depth(ref_e)) if deep_enough else nan
        metrics["pbe_vib_freq_error"] = (
            abs(vibrational_wavenumber(element, curvature) - vibrational_wavenumber(element, ref_curvature))
            if deep_enough
            else nan
        )
    # The shortest DFT separation can differ from 0.8 r_cov by one ulp.
    wall_min = evaluation_window(element, float(distances.max()), WALL_LOWER)[0] - 1e-12
    wall, ref_wall = (
        (distances >= wall_min) & (distances <= r_max),
        (ref_distances >= wall_min) & (ref_distances <= r_max),
    )
    if wall.sum() >= MIN_CURVE_POINTS and ref_wall.sum() >= MIN_CURVE_POINTS:
        metrics["pbe_wall_dist_mae"] = wall_error(
            ref_distances[ref_wall], ref_energies[ref_wall], distances[wall], energies[wall]
        )
    if ref_forces.size and len(ref_forces) == len(ref_distances) and overlap:
        metrics["pbe_force_mae"] = force_mae(ref_x, ref_forces[ref_window], x, forces[window])
    return metrics


def vib_freq_eligible(element: str, distances: ArrayLike, energies: ArrayLike) -> bool:
    """Whether a (smooth) reference curve has a vibrational wavenumber: a bound, convex well in the window."""
    distances, energies = np.asarray(distances, dtype=float), np.asarray(energies, dtype=float)
    r_min, r_max = evaluation_window(element, float(distances.max()))
    window = (distances >= r_min) & (distances <= r_max)
    x, e = distances[window], energies[window]
    if len(x) < MIN_FIT_POINTS or well_depth(e) < MIN_WELL_DEPTH:
        return False
    return bool(np.isfinite(vibrational_wavenumber(element, quadratic_well(x, e)[1])))
