"""The softening scale of a potential energy surface (B. Deng et al., npj Comput. Mater. 11, 9 (2025)).

On high-energy configurations, many MLIPs predict forces that are systematically smaller than DFT
forces: their PES is too soft. The softening scale is the slope ``a`` of the least-squares line
through the origin, ``F_MLIP ≈ a · F_DFT``, over all force components; ``a < 1`` means softening.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike


def softening_scale(dft_forces: Sequence[ArrayLike], mlip_forces: Sequence[ArrayLike]) -> float:
    """Slope of MLIP forces against DFT forces, a = Σ x·y / Σ x².

    This is the closed-form solution of the least-squares fit ``y = a x`` (upstream matcalc fits
    the same line with ``scipy.optimize.curve_fit``).

    Args:
        dft_forces: DFT forces of every frame, each of shape ``(n_atoms, 3)`` (eV/Å).
        mlip_forces: MLIP forces of the same frames, same shapes (eV/Å).

    Returns:
        The softening scale (dimensionless).
    """
    x = np.concatenate([np.asarray(forces, dtype=float).ravel() for forces in dft_forces])
    y = np.concatenate([np.asarray(forces, dtype=float).ravel() for forces in mlip_forces])
    return float(np.dot(x, y) / np.dot(x, x))
