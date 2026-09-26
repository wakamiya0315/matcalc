"""The MACE test model of the validation runs (not part of matcalc).

matcalc benchmarks whatever ASE calculator or TorchSim model the user passes. The validation runs of this
fork use MACE-MatPES-PBE-0; this module builds it for both paths from the same checkpoint file, so the
ASE and the TorchSim runs evaluate exactly the same potential. Needs mace-torch (and torch-sim).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

MODEL = "mace-matpes-pbe-0"


def load_mace(backend: Literal["ase", "torchsim"], *, dtype: str = "float64", device: str | None = None) -> Any:
    """MACE-MatPES-PBE-0 as an ASE calculator or as a TorchSim model.

    Args:
        backend: ``"ase"`` or ``"torchsim"``.
        dtype: ``"float64"`` or ``"float32"``.
        device: ``"cuda"`` or ``"cpu"`` (default: CUDA when available).

    Returns:
        The calculator or the TorchSim model.
    """
    from mace.calculators import mace_mp
    from mace.calculators.foundations_models import download_mace_mp_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "ase":
        return mace_mp(model=MODEL, device=device, default_dtype=dtype)
    from torch_sim.models.mace import MaceModel

    return MaceModel(
        model=download_mace_mp_checkpoint(MODEL),
        device=torch.device(device),
        dtype=getattr(torch, dtype),
        neighbor_list_fn=GrowingNeighborList() if device == "cuda" else None,
    )


class GrowingNeighborList:
    """Batched GPU neighbour list whose per-atom neighbour cap grows when a structure needs more room.

    TorchSim's default GPU neighbour list (nvalchemiops, naive N^2 per structure) sizes its buffers for at
    most 192 neighbours per atom within the cutoff, i.e. an average density of 0.2 atoms/Å^3; dense
    structures in the benchmark datasets have more and make it fail (TorchSim issue #545). This is the
    same algorithm with an explicit cap that is doubled until every structure fits and then kept.

    Attributes:
        max_neighbors: Current neighbour cap per atom.
    """

    def __init__(self, max_neighbors: int = 192) -> None:
        """
        Args:
            max_neighbors: Initial neighbour cap per atom.
        """
        self.max_neighbors = max_neighbors

    def __call__(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
        cutoff: float,
        system_idx: torch.Tensor,
        self_interaction: bool = False,  # noqa: FBT001, FBT002 - TorchSim's neighbour-list signature
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Neighbour pairs within ``cutoff`` for a batch of structures.

        Args:
            positions: Atomic positions, shape (n_atoms, 3) (Å).
            cell: Cells as row vectors, shape (n_structures, 3, 3) (Å).
            pbc: Periodic boundary conditions.
            cutoff: Cutoff radius (Å).
            system_idx: Structure index of every atom.
            self_interaction: Must be False (MACE does not use self pairs).

        Returns:
            Pairs (2, n_pairs), structure index of every pair, and the lattice shifts of every pair.

        Raises:
            ValueError: If ``self_interaction`` is requested.
        """
        from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError
        from nvalchemiops.torch.neighbors import batch_naive_neighbor_list
        from torch_sim.neighbors.utils import normalize_inputs

        if self_interaction:
            raise ValueError("GrowingNeighborList does not produce self pairs")
        cell, pbc = normalize_inputs(cell, pbc, int(system_idx.max().item()) + 1)
        while True:
            try:
                result = batch_naive_neighbor_list(
                    positions=positions,
                    cutoff=float(cutoff),
                    batch_idx=system_idx.to(torch.int32),
                    cell=cell,
                    pbc=pbc.to(torch.bool),
                    max_neighbors=self.max_neighbors,
                    return_neighbor_list=True,
                )
                break
            except NeighborOverflowError:
                self.max_neighbors *= 2
                logger.info("Neighbour cap raised to %d per atom", self.max_neighbors)
        pairs = result[0].to(torch.long)
        shifts = result[2] if len(result) == 3 else torch.zeros((pairs.shape[1], 3), device=positions.device)  # noqa: PLR2004
        return pairs, system_idx[pairs[0]], shifts.to(cell.dtype)


