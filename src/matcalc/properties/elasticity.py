"""Elastic constants from stress-strain fits (the matcalc / pymatgen protocol).

The relaxed cell is strained along each of the six Voigt components (xx, yy, zz, yz, xz, xy) with
four small strains each, and the stress of every strained cell is computed. Each elastic constant
C_ij is the slope of a straight line fitted to stress component j against strain component i.
The bulk (K) and shear (G) moduli are the Voigt-Reuss-Hill (VRH) averages of the tensor C.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pymatgen.core.elasticity import DeformedStructureSet, ElasticTensor, Strain
from pymatgen.core.elasticity.elastic import get_strain_state_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

NORMAL_STRAINS: tuple[float, ...] = (-0.01, -0.005, 0.005, 0.01)
"""Strains applied along xx, yy and zz (dimensionless)."""

SHEAR_STRAINS: tuple[float, ...] = (-0.06, -0.03, 0.03, 0.06)
"""Strains applied along yz, xz and xy (dimensionless)."""

ZERO_TOLERANCE = 1e-7
"""Fitted constants smaller than this (eV/Å^3) are set to zero, as upstream matcalc does."""


def strained_structures(
    structure: Structure,
    normal_strains: Sequence[float] = NORMAL_STRAINS,
    shear_strains: Sequence[float] = SHEAR_STRAINS,
) -> tuple[list[Structure], list[Strain]]:
    """Strain the relaxed cell along each Voigt component.

    Args:
        structure: Relaxed structure.
        normal_strains: Strains along xx, yy and zz.
        shear_strains: Strains along yz, xz and xy.

    Returns:
        The strained structures (6 x 4 = 24 with the defaults) and the Green-Lagrange strain of each.
    """
    deformed = DeformedStructureSet(structure, normal_strains, shear_strains, symmetry=False)
    strains = [Strain.from_deformation(deformation) for deformation in deformed.deformations]
    return list(deformed), strains


@dataclass
class ElasticFit:
    """Fitted elastic tensor and moduli, all in GPa.

    Attributes:
        elastic_tensor: The 6x6 Voigt matrix C_ij (GPa).
        bulk_modulus_vrh: Bulk modulus K, Voigt-Reuss-Hill average (GPa).
        shear_modulus_vrh: Shear modulus G, Voigt-Reuss-Hill average (GPa).
        youngs_modulus: Young's modulus from K and G, 9KG / (3K + G) (GPa).
        mean_r2: Mean coefficient of determination of the linear fits (1 = perfectly linear).
        residuals_sum: Sum of squared residuals of all fits (GPa).
    """

    elastic_tensor: np.ndarray
    bulk_modulus_vrh: float
    shear_modulus_vrh: float
    youngs_modulus: float
    mean_r2: float
    residuals_sum: float


def fit_elastic_tensor(
    strains: Sequence[Strain],
    stresses: Sequence[np.ndarray],
    equilibrium_stress: np.ndarray,
) -> ElasticFit:
    """Fit the elastic tensor to stress-strain data.

    Args:
        strains: Strain of every strained cell (from ``strained_structures``).
        stresses: Stress tensor of every strained cell, same order (3x3, eV/Å^3, ASE sign).
        equilibrium_stress: Stress of the unstrained relaxed cell (eV/Å^3); it is included as the
            zero-strain point of every fit, so a small residual stress does not bias the slopes.

    Returns:
        The elastic tensor and the moduli in GPa.
    """
    voigt_directions = [tuple(direction) for direction in np.eye(6)]
    by_direction = get_strain_state_dict(strains, stresses, eq_stress=equilibrium_stress, add_eq=True)
    c_ij = np.zeros((6, 6))
    residuals_sum = 0.0
    r2_values: list[float] = []
    for i in range(6):
        strain = by_direction[voigt_directions[i]]["strains"]
        stress = by_direction[voigt_directions[i]]["stresses"]
        for j in range(6):
            x, y = strain[:, i], stress[:, j]
            coefficients, residuals, *_ = np.polyfit(x, y, 1, full=True)
            c_ij[i, j] = coefficients[0]
            residual = residuals[0] if len(residuals) > 0 else 0.0
            residuals_sum += residual
            total = float(np.sum((y - np.mean(y)) ** 2))
            if total > 0:  # components that are zero by symmetry carry no information
                r2_values.append(1.0 - float(residual) / total)

    tensor = ElasticTensor.from_voigt(c_ij).zeroed(ZERO_TOLERANCE)
    to_gpa = 1 / tensor.GPa_to_eV_A3
    bulk, shear = tensor.k_vrh, tensor.g_vrh
    # Young's modulus from the same VRH averages; pymatgen's y_mod assumes K and G in GPa.
    youngs = 9 * bulk * shear / (3 * bulk + shear)
    return ElasticFit(
        elastic_tensor=np.asarray(tensor.voigt) * to_gpa,
        bulk_modulus_vrh=float(bulk * to_gpa),
        shear_modulus_vrh=float(shear * to_gpa),
        youngs_modulus=float(youngs * to_gpa),
        mean_r2=float(np.mean(r2_values)) if r2_values else 1.0,
        residuals_sum=float(residuals_sum * to_gpa),
    )
