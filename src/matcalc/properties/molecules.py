"""Energies of molecules and molecular complexes against quantum-chemical references, in kcal/mol.

Three quantities, defined as in MLIPAudit (InstaDeep, arXiv:2511.20487), whose published results for several
MLIPs they reproduce:

- The interaction energy of a complex from a dissociation curve (energies at contact distances scaled from
  0.8 to 2.0 times the equilibrium one): the lowest energy of the curve (the highest for repulsive contacts)
  minus the energy at the largest separation.
- The relative energies of the conformers of a molecule, with the conformer lowest in the reference as the
  zero of both the reference and the prediction: their mean absolute and root-mean-square errors, and the
  Spearman rank correlation of the two.
- The barrier height and the reaction energy of an elementary reaction: the energy of the transition state
  and the summed energy of the products, each minus the energy of the reactants (electronic energies, no
  zero-point energy).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase import units

if TYPE_CHECKING:
    from collections.abc import Sequence

KCAL_PER_MOL = units.kcal / units.mol
"""1 kcal/mol in eV."""


def interaction_energy(
    distances: Sequence[float] | np.ndarray, energies: Sequence[float] | np.ndarray, *, repulsive: bool = False
) -> float:
    """Depth of the well of a dissociation curve (or height of the wall for repulsive contacts).

    Args:
        distances: A measure of the separation at every point (any unit; only its largest value matters).
        energies: Energies at the points.
        repulsive: Take the highest energy instead of the lowest.

    Returns:
        The extreme energy of the curve minus the energy at the largest separation (unit of ``energies``).
    """
    values = np.asarray(energies, dtype=float)
    separated = values[int(np.argmax(distances))]
    return float((values.max() if repulsive else values.min()) - separated)


def conformer_errors(
    reference: Sequence[float] | np.ndarray, predicted: Sequence[float] | np.ndarray
) -> dict[str, float]:
    """Errors of the relative energies of a molecule's conformers.

    Args:
        reference: Reference energies of the conformers (kcal/mol, any zero).
        predicted: Predicted energies of the same conformers (kcal/mol, any zero).

    Returns:
        ``mae`` and ``rmse`` (kcal/mol) of the energies relative to the conformer lowest in the reference,
        and ``spearman``, the rank correlation of the two sets of relative energies.
    """
    from scipy.stats import spearmanr

    ref, pred = np.asarray(reference, dtype=float), np.asarray(predicted, dtype=float)
    lowest = int(np.argmin(ref))
    ref, pred = ref - ref[lowest], pred - pred[lowest]
    errors = pred - ref
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "spearman": float(spearmanr(ref, pred).statistic),
    }


def reaction_energetics(
    reactant: float, transition_state: float, products: Sequence[float] | np.ndarray
) -> dict[str, float]:
    """Barrier height and reaction energy of an elementary reaction.

    Args:
        reactant: Energy of the reactant (kcal/mol).
        transition_state: Energy of the transition state (kcal/mol).
        products: Energies of the product molecules, each on its own (kcal/mol).

    Returns:
        ``barrier``, the transition state minus the reactant, and ``reaction_energy``, the products minus the
        reactant (kcal/mol).
    """
    return {"barrier": transition_state - reactant, "reaction_energy": float(np.sum(products)) - reactant}


def error_statistics(deviations: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Summary of the deviations of predictions from their references.

    Args:
        deviations: Prediction minus reference for every system.

    Returns:
        ``MAE``, ``RMSE`` and ``ME`` (the mean signed deviation), NaN without deviations, and their number ``n``.
    """
    values = np.asarray(deviations, dtype=float)
    if not values.size:
        return {"MAE": float("nan"), "RMSE": float("nan"), "ME": float("nan"), "n": 0}
    return {
        "MAE": float(np.mean(np.abs(values))),
        "RMSE": float(np.sqrt(np.mean(values**2))),
        "ME": float(np.mean(values)),
        "n": int(values.size),
    }
