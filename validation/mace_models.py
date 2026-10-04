"""The MACE test models of the validation runs (not part of matcalc).

matcalc benchmarks whatever ASE calculator or TorchSim model the user passes. The validation runs of this
fork use MACE-MatPES-PBE-0 (MACE-MP-0 and MACE-OFF23 where published results of those models exist; MACE-OMAT-0
and MACE-MH-1 for the Adsorption benchmark); this module builds them for both paths from the same checkpoint
file, so the ASE and the TorchSim runs evaluate exactly the same potential, optionally with a D3 dispersion
correction. Needs mace-torch (and torch-sim; torch-dftd for D3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

MODEL = "mace-matpes-pbe-0"
"""Default test model. ``"medium"`` is MACE-MP-0 (medium, trained on MPtrj): the model whose Matbench
Discovery results are published, used to validate the Discovery benchmark. ``"off-medium"`` is MACE-OFF23
(medium; organic molecules), whose MLIPAudit results validate the molecular benchmarks."""

MACE_OFF_ELEMENTS = ("H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I")
"""The elements of MACE-OFF23."""

D3_BJ = {"pbe": {"a1": 0.4289, "s8": 0.7875, "a2": 4.4407}, "rpbe": {"a1": 0.1820, "s8": 0.8318, "a2": 4.0094}}
"""Grimme's D3(BJ) damping parameters (``a2`` in Bohr) of PBE and RPBE, for TorchSim's ``D3DispersionModel``
(torch-dftd has the same)."""


def load_mace(
    backend: Literal["ase", "torchsim"],
    *,
    model: str = MODEL,
    dtype: str = "float64",
    device: str | None = None,
    head: str | None = None,
    d3: str | None = None,
) -> Any:
    """A MACE foundation model (default MACE-MatPES-PBE-0) as an ASE calculator or as a TorchSim model.

    Args:
        backend: ``"ase"`` or ``"torchsim"``.
        model: Name of the checkpoint for ``mace_mp`` (e.g. ``"mace-matpes-pbe-0"``, ``"medium"``,
            ``"medium-omat-0"``, ``"mh-1"``), or ``"off-<size>"`` for MACE-OFF23 (``"off-small"``,
            ``"off-medium"``, ``"off-large"``).
        dtype: ``"float64"`` or ``"float32"``.
        device: ``"cuda"`` or ``"cpu"`` (default: CUDA when available).
        head: Head of a multi-head model (e.g. ``"oc20_usemppbe"`` of MACE-MH-1); default: the model's own.
        d3: Add Grimme's D3(BJ) dispersion correction with the damping parameters of this functional
            (``"pbe"``, ``"rpbe"``): torch-dftd's ``TorchDFTD3Calculator`` (its default cutoffs) for ASE,
            TorchSim's ``D3DispersionModel`` with torch-dftd's reference parameters for TorchSim.

    Returns:
        The calculator or the TorchSim model.
    """
    from mace.calculators import mace_mp, mace_off
    from mace.calculators.foundations_models import download_mace_mp_checkpoint, mace_off_urls
    from mace.tools.utils import get_cache_dir

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    off_size = model.removeprefix("off-") if model.startswith("off-") else None
    heads = {} if head is None else {"head": head}
    if backend == "ase":
        if off_size is not None:
            calculator = mace_off(model=off_size, device=device, default_dtype=dtype)
        else:
            calculator = mace_mp(model=model, device=device, default_dtype=dtype, **heads)
        if not d3:
            return calculator
        from ase.calculators.mixing import SumCalculator

        return SumCalculator([calculator, ase_d3(d3, device=device, dtype=dtype)])
    from torch_sim.models.mace import MaceModel

    if off_size is not None:
        checkpoint = Path(get_cache_dir()) / Path(mace_off_urls[off_size]).name
        if not checkpoint.exists():
            mace_off(model=off_size, device="cpu")  # downloads it into MACE's cache
    else:
        checkpoint = Path(download_mace_mp_checkpoint(model))
    mace = MaceModel(
        model=str(checkpoint),
        device=torch.device(device),
        dtype=getattr(torch, dtype),
        neighbor_list_fn=GrowingNeighborList() if device == "cuda" else None,
        **heads,
    )
    if not d3:
        return mace
    from torch_sim.models.interface import SumModel

    return SumModel(mace, torchsim_d3(d3, device=device, dtype=dtype))


def ase_d3(functional: str = "pbe", *, device: str, dtype: str) -> Any:
    """torch-dftd's D3(BJ) correction of a functional as an ASE calculator, with its default cutoffs (95 Bohr;
    40 Bohr for the coordination numbers).
    """
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    return TorchDFTD3Calculator(device=device, damping="bj", xc=functional, dtype=getattr(torch, dtype))


def torchsim_d3(functional: str = "pbe", *, device: str, dtype: str) -> Any:
    """TorchSim's D3(BJ) correction of a functional (``D3DispersionModel``), with torch-dftd's reference
    parameters.

    torch-dftd keeps the C6 coefficients and the coordination numbers of the reference pairs in one array (C6,
    CN of the first, CN of the second atom); the coordination numbers of the second atom are those of the
    first with the pair swapped, so the first two are all TorchSim needs. TorchSim counts neighbours for the
    coordination numbers out to the D3 cutoff (95 Bohr), torch-dftd out to 40 Bohr: the two give the same
    dispersion energies within 0.3 meV per adsorbed slab of the Adsorption benchmark, within 6e-5 eV with
    torch-dftd's ``cnthr`` raised to 95 Bohr.
    """
    import numpy as np
    import torch_dftd
    from nvalchemiops.torch.interactions.dispersion import D3Parameters
    from torch_sim.models.dispersion import D3DispersionModel

    raw = np.load(Path(torch_dftd.__file__).parent / "nn" / "params" / "dftd3_params.npz")
    parameters = D3Parameters(
        rcov=torch.tensor(raw["rcov"]),
        r4r2=torch.tensor(raw["r2r4"]),
        c6ab=torch.tensor(raw["c6ab"][..., 0]),
        cn_ref=torch.tensor(raw["c6ab"][..., 1]),
    ).to(device=device)
    return D3DispersionModel(
        **D3_BJ[functional], d3_params=parameters, device=torch.device(device), dtype=getattr(torch, dtype)
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


