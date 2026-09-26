"""Load MACE foundation models by name, for ASE or for TorchSim.

Any ASE calculator (or TorchSim model) can be benchmarked; this module only makes the MACE models used
for validating this fork one call away. Both backends load the same checkpoint file, so the ASE and the
TorchSim runs of a benchmark evaluate exactly the same potential.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

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
    backend: Literal["ase", "torchsim"] = "ase",
    device: str | None = None,
    dtype: Literal["float64", "float32"] = "float64",
    cueq: bool = False,
) -> Any:
    """Load a MACE foundation model.

    Args:
        name: A key of ``MACE_MODELS``, any model name accepted by ``mace_mp``, or a path to a
            ``.model`` file. The default, MACE-MatPES-PBE-0, is trained on MatPES (PBE) data and is
            distributed under the Academic Software License.
        backend: ``"ase"`` for an ASE calculator (use with ``ASESimulator``), ``"torchsim"`` for a
            TorchSim model (use with ``TorchSimSimulator``; needs ``torch-sim-atomistic``).
        device: ``"cuda"`` or ``"cpu"``; default: CUDA when available.
        dtype: Floating-point precision of the model. float64 is the default because finite
            displacements (phonons) and small strains (elasticity) need it.
        cueq: Use NVIDIA cuEquivariance kernels for the tensor products (needs the
            ``cuequivariance-torch`` and ``cuequivariance-ops-torch-cu12`` packages and a CUDA GPU).
            With float32 they compute the forces on phonon supercells 10-20x faster than the default
            kernels; with float64 they are slower (docs/validation.md).

    Returns:
        The MACE ASE calculator, or the MACE TorchSim model.
    """
    import torch
    from mace.calculators import mace_mp
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MACE_MODELS.get(name, name)
    if backend == "ase":
        return mace_mp(model=model, device=device, default_dtype=dtype, enable_cueq=cueq)

    from torch_sim.models.mace import MaceModel

    from matcalc.simulation.torchsim import GrowingNeighborList

    checkpoint = model if Path(model).is_file() else download_mace_mp_checkpoint(model)
    return MaceModel(
        model=checkpoint,
        device=torch.device(device),
        dtype=getattr(torch, dtype),
        enable_cueq=cueq,
        neighbor_list_fn=GrowingNeighborList() if device == "cuda" else None,
    )
