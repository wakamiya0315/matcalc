"""Load MACE foundation models by name.

Any ASE calculator can be benchmarked; this module only makes the MACE models used for validating
this fork one call away.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ase.calculators.calculator import Calculator

MACE_MODELS: dict[str, str] = {
    "MACE-MatPES-PBE-0": "mace-matpes-pbe-0",
    "MACE-MatPES-r2SCAN-0": "mace-matpes-r2scan-0",
    "MACE-MPA-0-medium": "medium-mpa-0",
    "MACE-OMAT-0-medium": "medium-omat-0",
    "MACE-MP-0-medium": "medium",
}
"""Readable model names → the names ``mace.calculators.mace_mp`` understands."""


def load_mace(
    name: str = "MACE-MatPES-PBE-0",
    *,
    device: str | None = None,
    dtype: Literal["float64", "float32"] = "float64",
) -> Calculator:
    """Load a MACE foundation model as an ASE calculator.

    Args:
        name: A key of ``MACE_MODELS``, any model name accepted by ``mace_mp``, or a path to a
            ``.model`` file. The default, MACE-MatPES-PBE-0, is trained on MatPES (PBE) data and is
            distributed under the Academic Software License.
        device: ``"cuda"`` or ``"cpu"``; default: CUDA when available.
        dtype: Floating-point precision of the model. float64 is the default because finite
            displacements (phonons) and small strains (elasticity) need it.

    Returns:
        The MACE ASE calculator.
    """
    import torch
    from mace.calculators import mace_mp

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return mace_mp(model=MACE_MODELS.get(name, name), device=device, default_dtype=dtype)
