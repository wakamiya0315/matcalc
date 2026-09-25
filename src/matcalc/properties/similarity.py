"""Structural distance between an MLIP-relaxed structure and the DFT-relaxed one.

Each structure is described by a vector: statistics (mean, standard deviation, minimum, maximum)
over its sites of the CrystalNN local-environment fingerprint (matminer's ``SiteStatsFingerprint``,
"ops" preset). The distance ``d`` is the Euclidean norm of the difference of the two vectors;
``d = 0`` means identical local environments.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pymatgen.core import Structure


@cache
def _fingerprinter() -> Any:
    # matminer is slow to import and only needed here, so it is imported on first use.
    from matminer.featurizers.site import CrystalNNFingerprint
    from matminer.featurizers.structure import SiteStatsFingerprint

    return SiteStatsFingerprint(
        CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
        stats=("mean", "std_dev", "minimum", "maximum"),
    )


def structure_fingerprint(structure: Structure) -> np.ndarray:
    """Local-environment fingerprint of a structure.

    Args:
        structure: The structure.

    Returns:
        The fingerprint vector.
    """
    return np.array(_fingerprinter().featurize(structure))


def fingerprint_distance(fingerprint_a: np.ndarray, fingerprint_b: np.ndarray) -> float:
    """Euclidean distance between two fingerprints.

    Args:
        fingerprint_a: Fingerprint of the first structure.
        fingerprint_b: Fingerprint of the second structure.

    Returns:
        The distance ``d`` (dimensionless).
    """
    return float(np.linalg.norm(fingerprint_a - fingerprint_b))


def structure_fingerprint_or_error(structure: Structure) -> np.ndarray | str:
    """``structure_fingerprint``, or the error message if it cannot be computed (for batch use).

    Args:
        structure: The structure.

    Returns:
        The fingerprint vector, or ``"<ErrorType>: <message>"``.
    """
    try:
        return structure_fingerprint(structure)
    except Exception as exc:  # noqa: BLE001 - one odd structure must not stop the others
        return f"{type(exc).__name__}: {exc}"
