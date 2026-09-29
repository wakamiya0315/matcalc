"""Thermodynamic stability of crystals from their distance to the convex hull, and relaxed geometries.

The definitions follow Matbench Discovery (J. Riebesell et al., Nat. Mach. Intell. 7, 836 (2025)):

- A crystal counts as stable when its energy above the convex hull of the known phases is at most 0.
  The MLIP's hull distance is the DFT one shifted by the error of the formation energy (the hull itself,
  built from Materials Project entries, stays fixed).
- Classification metrics count true and false positives among the crystals called stable; the
  discovery acceleration factor (DAF) is the precision divided by the fraction of stable crystals, i.e.
  how much more often a crystal picked by the model is stable than one picked at random.
- A relaxed geometry is compared with the DFT one by the root-mean-square displacement of pymatgen's
  ``StructureMatcher`` (normalized by (volume per atom)^(1/3)) and by the space group moyopy finds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatgen.core import Structure

STABILITY_THRESHOLD = 0.0
"""Largest energy above the convex hull (eV/atom) of a crystal that counts as stable."""

SYMPRECS: tuple[float, ...] = (1e-5, 1e-2)
"""Symmetry tolerances (Å) at which the space groups of relaxed and DFT structures are compared."""

UNMATCHED_RMSD = 1.0
"""RMSD given to a structure that ``StructureMatcher`` cannot match with its DFT counterpart (its ``stol``)."""


def stability_metrics(
    e_above_hull_true: Sequence[float] | np.ndarray,
    e_above_hull_pred: Sequence[float] | np.ndarray,
    *,
    prevalence: float | None = None,
    threshold: float = STABILITY_THRESHOLD,
) -> dict[str, float]:
    """Classification and regression metrics of predicted distances to the convex hull.

    Crystals without a prediction (NaN) count as predicted unstable; the regression metrics leave them
    out.

    Args:
        e_above_hull_true: DFT energies above the hull (eV/atom).
        e_above_hull_pred: Predicted energies above the hull (eV/atom), NaN where missing.
        prevalence: Fraction of stable crystals used for the DAF; default: that of ``e_above_hull_true``.
        threshold: Stability threshold (eV/atom).

    Returns:
        F1, DAF, Precision, Recall, Accuracy, TP, FP, TN, FN, MAE, RMSE, R2 (energies in eV/atom) and the
        number of missing predictions.
    """
    true = np.asarray(e_above_hull_true, dtype=float)
    pred = np.asarray(e_above_hull_pred, dtype=float)
    actual_stable = true <= threshold
    predicted_stable = np.nan_to_num(pred, nan=np.inf) <= threshold
    tp = int(np.sum(actual_stable & predicted_stable))
    fn = int(np.sum(actual_stable & ~predicted_stable))
    fp = int(np.sum(~actual_stable & predicted_stable))
    tn = int(np.sum(~actual_stable & ~predicted_stable))
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if prevalence is None:
        prevalence = _ratio(tp + fn, len(true))
    present = ~np.isnan(pred)
    errors = pred[present] - true[present]
    total = float(np.sum((true[present] - true[present].mean()) ** 2))
    return {
        "F1": _ratio(2 * precision * recall, precision + recall),
        "DAF": _ratio(precision, prevalence),
        "Precision": precision,
        "Recall": recall,
        "Accuracy": _ratio(tp + tn, len(true)),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "MAE": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
        "RMSE": float(np.sqrt(np.mean(errors**2))) if errors.size else float("nan"),
        "R2": 1.0 - float(np.sum(errors**2)) / total if errors.size > 1 and total > 0 else float("nan"),
        "missing": int(np.sum(~present)),
    }


@dataclass
class GeometryComparison:
    """A relaxed structure compared with its DFT counterpart.

    Attributes:
        rmsd: RMS displacement after matching, normalized by (volume per atom)^(1/3); NaN if the structures
            do not match.
        space_groups: Space group number of the relaxed structure at each tolerance of ``SYMPRECS``.
        n_symmetry_operations: Number of symmetry operations at each tolerance.
    """

    rmsd: float
    space_groups: tuple[int, ...]
    n_symmetry_operations: tuple[int, ...]


def compare_geometry(relaxed: Structure, reference: Structure) -> GeometryComparison:
    """RMSD from the DFT structure (pymatgen ``StructureMatcher``) and symmetry (moyopy) of a relaxed structure.

    Args:
        relaxed: Structure relaxed with the MLIP.
        reference: DFT-relaxed structure.

    Returns:
        The comparison.
    """
    import moyopy
    from moyopy.interface import MoyoAdapter
    from pymatgen.analysis.structure_matcher import StructureMatcher

    match = StructureMatcher(stol=UNMATCHED_RMSD, scale=False).get_rms_dist(relaxed, reference)
    cell = MoyoAdapter.from_py_obj(relaxed)
    datasets = [moyopy.MoyoDataset(cell, symprec=symprec) for symprec in SYMPRECS]
    return GeometryComparison(
        rmsd=float(match[0]) if match is not None else float("nan"),
        space_groups=tuple(int(d.number) for d in datasets),
        n_symmetry_operations=tuple(int(d.operations.num_operations) for d in datasets),
    )


def geometry_metrics(
    rmsd: Sequence[float] | np.ndarray,
    space_groups: Sequence[float] | np.ndarray,
    reference_space_groups: Sequence[float] | np.ndarray,
    n_operations: Sequence[float] | np.ndarray,
    reference_n_operations: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """How closely relaxed structures reproduce the DFT geometries.

    Args:
        rmsd: RMSD of every relaxed structure (NaN where unmatched: counted as ``UNMATCHED_RMSD``).
        space_groups: Space group numbers of the relaxed structures (NaN where unknown).
        reference_space_groups: Those of the DFT structures.
        n_operations: Numbers of symmetry operations of the relaxed structures.
        reference_n_operations: Those of the DFT structures.

    Returns:
        Mean RMSD, mean absolute error of the number of symmetry operations, fractions of structures whose
        symmetry decreased, matched and increased, and the number of structures with known symmetry.
    """
    distances = np.nan_to_num(np.asarray(rmsd, dtype=float), nan=UNMATCHED_RMSD)
    spg_change = np.asarray(space_groups, dtype=float) - np.asarray(reference_space_groups, dtype=float)
    ops_change = np.asarray(n_operations, dtype=float) - np.asarray(reference_n_operations, dtype=float)
    known = ~np.isnan(spg_change)
    changed = known & (spg_change != 0)
    n_known = int(np.sum(known))
    return {
        "rmsd": float(np.mean(distances)) if distances.size else float("nan"),
        "n_sym_ops_mae": float(np.mean(np.abs(ops_change[known]))) if n_known else float("nan"),
        "symmetry_decrease": _ratio(int(np.sum(changed & (ops_change < 0))), n_known),
        "symmetry_match": _ratio(int(np.sum(known & ~changed)), n_known),
        "symmetry_increase": _ratio(int(np.sum(changed & (ops_change > 0))), n_known),
        "n_structures": n_known,
    }


def mp2020_correction_change(relaxed: Structure, reference: Structure, energy: float, parameters: dict) -> float | None:
    """Change of the MP2020 energy correction of an entry when its structure is replaced by the relaxed one.

    The MP2020 corrections depend on the structure only through the oxide type (oxide, peroxide, superoxide,
    ozonide or hydroxide, from the O-O and O-H distances) and the sulfide type (sulfide, polysulfide or
    sulfate, from the neighbors of S). Matbench Discovery corrects the MLIP-relaxed structure; computing the
    correction for both structures with the same pymatgen and adding the difference to the published DFT
    correction gives the same result without depending on how the pymatgen version assigns the other
    corrections (matbench-discovery issue #358).

    Args:
        relaxed: Structure relaxed with the MLIP.
        reference: DFT-relaxed structure of the entry.
        energy: DFT energy of the entry (eV); the correction does not depend on it.
        parameters: ``parameters`` of the DFT ``ComputedStructureEntry`` (run type, Hubbard U, POTCARs).

    Returns:
        Correction for the relaxed structure minus that for the DFT structure (eV per cell): 0 if both have
        the same oxide and sulfide types, ``None`` if pymatgen rejects either entry.
    """
    import warnings

    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility, oxide_type, sulfide_type
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    # The relative cutoff of the O-O and O-H distances that MaterialsProject2020Compatibility uses
    if (oxide_type(relaxed, 1.05), sulfide_type(relaxed)) == (oxide_type(reference, 1.05), sulfide_type(reference)):
        return 0.0
    compatibility = MaterialsProject2020Compatibility()
    corrections = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "Failed to guess oxidation states", as for the DFT entries
        for structure in (relaxed, reference):
            entry = ComputedStructureEntry(structure, energy, parameters=parameters, data={})
            processed = compatibility.process_entry(entry, clean=True)
            if processed is None:
                return None
            corrections.append(processed.correction)
    return float(corrections[0] - corrections[1])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")
