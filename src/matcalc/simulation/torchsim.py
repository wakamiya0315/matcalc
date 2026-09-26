"""Batched relaxations and single points on a GPU with TorchSim.

``ASESimulator`` sends one small cell at a time to the GPU, which leaves it mostly idle.
``TorchSimSimulator`` evaluates many structures in each forward pass instead: the structures of a batch
are concatenated into one graph, and batches are sized to fill the GPU memory.

- Relaxations use in-flight batching: when a structure has relaxed it leaves the batch and the next one
  takes its place. The optimizer is TorchSim's ASE-flavoured FIRE on a Frechet cell filter, the same
  algorithm as ``ASESimulator`` (ASE's FIRE on a ``FrechetCellFilter``).
- Single points are packed into batches by size.

TorchSim needs the MLIP as a TorchSim model (``torch_sim.models.interface.ModelInterface``), for example
``matcalc.load_mace(backend="torchsim")``.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch_sim as ts
import torch_sim.math as tsm
from torch_sim.autobatching import calculate_memory_scalers, estimate_max_memory_scaler, to_constant_volume_bins
from torch_sim.optimizers import fire_init, fire_step
from tqdm import tqdm

from matcalc.structures import to_ase_atoms, to_pmg_structure

from .base import RelaxResult, SinglePointResult

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from ase import Atoms
    from pymatgen.core import Structure
    from torch_sim.models.interface import ModelInterface

logger = logging.getLogger(__name__)

FIRE_DEFAULTS = {"n_min": 5, "f_dec": 0.5, "f_alpha": 0.99}
"""FIRE parameters used below; the defaults of both TorchSim and ASE."""

MAX_OUT_OF_MEMORY_RETRIES = 3
"""How often a batched call is retried with half the batch capacity after running out of GPU memory."""

OUT_OF_MEMORY_BACKOFF = 0.8
"""After a batch of single points runs out of GPU memory, the capacity is lowered to this fraction of it."""

MAX_PROBE_ATOMS = 100_000
"""Largest batch (in atoms) tried when measuring the batch capacity. The GPU is saturated well before; a
model that needs little memory per atom (float32 with cuEquivariance) would otherwise be probed with
batches of up to half a million atoms, which takes minutes."""

OUT_OF_MEMORY_MESSAGES = ("out of memory", "Failed to allocate")
"""Parts of the messages of out-of-memory errors: PyTorch's, and those of kernels that allocate GPU memory
themselves, such as cuEquivariance's ("cudaErrorMemoryAllocation:out of memory")."""


def ase_consistent_fire_step(state: Any, model: ModelInterface, **kwargs: Any) -> Any:
    """One step of TorchSim's ASE-flavoured FIRE, corrected to follow ASE's FIRE exactly.

    Two details of TorchSim's step differ from ``ase.optimize.FIRE``; both are undone here, so every
    structure follows the same path as in ASE whatever batch it happens to be in:

    1. A structure that has not moved yet has NaN velocities. When a whole batch is new, TorchSim skips
       the velocity mixing like ASE's first step, but a structure that joins a running batch goes
       through the mixing with zero power, which halves its time step. Its time step is doubled
       beforehand.
    2. After more than ``n_min`` downhill steps FIRE shrinks the mixing parameter alpha. ASE mixes the
       velocities with alpha from *before* shrinking it; TorchSim shrinks it first. For the structures
       concerned, alpha is enlarged by the same factor before the step and shrunk again afterwards.

    Args:
        state: TorchSim FIRE state (with a cell filter) of the current batch.
        model: TorchSim model.
        **kwargs: FIRE parameters, passed on to ``torch_sim.optimizers.fire_step``.

    Returns:
        The state after one FIRE step.
    """
    n_min = kwargs.get("n_min", FIRE_DEFAULTS["n_min"])
    f_dec = kwargs.get("f_dec", FIRE_DEFAULTS["f_dec"])
    f_alpha = kwargs.get("f_alpha", FIRE_DEFAULTS["f_alpha"])
    new_atoms = state.velocities.isnan().any(dim=1)
    if new_atoms.all():  # first step of every structure: TorchSim already does what ASE does
        return fire_step(state, model, **kwargs)

    if new_atoms.any():
        new_structures = torch.zeros(state.n_systems, dtype=torch.bool, device=state.device)
        new_structures[state.system_idx[new_atoms]] = True
        state.dt[new_structures] = state.dt[new_structures] / f_dec

    # The power P = F . v that decides the step, computed as TorchSim does inside fire_step.
    power = tsm.batched_vdot(state.deform_grad_forces(), torch.nan_to_num(state.velocities), state.system_idx) + (
        state.cell_forces * torch.nan_to_num(state.cell_velocities)
    ).sum(dim=(1, 2))
    shrinking = (state.n_pos > n_min) & (power > 0.0)
    state.alpha[shrinking] = state.alpha[shrinking] / f_alpha
    state = fire_step(state, model, **kwargs)
    state.alpha[shrinking] = state.alpha[shrinking] * f_alpha
    return state


def ase_convergence(fmax: float) -> Callable[..., torch.Tensor]:
    """Convergence test of ASE's FIRE on a ``FrechetCellFilter``, for a batch of structures.

    ASE stops when every row of the filter's forces is below ``fmax``: the atomic forces transformed by
    the deformation gradient (``forces @ F``) and the three cell forces. TorchSim's own force criterion
    uses the untransformed atomic forces, which can stop a relaxation one step earlier or later.

    Args:
        fmax: Force threshold (eV/Å).

    Returns:
        A TorchSim convergence function: state → boolean tensor, one entry per structure.
    """

    def converged(state: Any, last_energy: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG001
        norms = state.deform_grad_forces().norm(dim=1)
        atom_max = torch.zeros(state.n_systems, device=state.device, dtype=state.dtype).scatter_reduce(
            0, state.system_idx, norms, reduce="amax"
        )
        cell_max = state.cell_forces.norm(dim=2).max(dim=1).values
        return (atom_max < fmax) & (cell_max < fmax)

    return converged


def converged_before_relaxing(structure: Structure | Atoms, start: SinglePointResult, fmax: float) -> bool:
    """ASE's convergence test before the first FIRE step, when the deformation gradient is the identity.

    The cell forces of a Frechet cell filter are then the virial (-volume x stress) divided by the number
    of atoms.
    ASE takes no step at all for such a structure, whereas TorchSim always takes at least one.

    Args:
        structure: The structure.
        start: Its single point (forces and stress).
        fmax: Force threshold (eV/Å).

    Returns:
        Whether ASE would stop before the first step.
    """
    if start.error is not None or start.forces is None or start.stress is None:
        return False
    atoms = to_ase_atoms(structure)
    cell_forces = -atoms.get_volume() * np.asarray(start.stress) / len(atoms)
    largest = max(np.linalg.norm(start.forces, axis=1).max(), np.linalg.norm(cell_forces, axis=1).max())
    return bool(largest < fmax)


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


class TorchSimSimulator:
    """Evaluate many structures per GPU forward pass with TorchSim.

    Attributes:
        model: TorchSim model of the MLIP.
        max_memory_scaler: Capacity of one batch in TorchSim's memory metric (sum over the batch of
            number of atoms x number density). ``None`` = measure it on the GPU for every call, on the
            smallest and the largest structure of that call.
        memory_padding: Fraction of the measured capacity actually used (TorchSim cannot recover
            from running out of GPU memory in the middle of a run).
        capacities: Capacity used by every call so far (for diagnostics).
        steps_between_swaps: FIRE steps between convergence checks. 1 stops each relaxation at the
            same step as ASE; larger values do less bookkeeping.
        show_progress: Show progress bars.
    """

    batched = True
    """Benchmarks give batched simulators large chunks of materials."""

    def __init__(
        self,
        model: ModelInterface,
        *,
        max_memory_scaler: float | None = None,
        memory_padding: float = 0.9,
        steps_between_swaps: int = 1,
        show_progress: bool = True,
    ) -> None:
        """
        Args:
            model: TorchSim model of the MLIP (its device and dtype are used for everything).
            max_memory_scaler: Capacity of one batch in TorchSim's memory metric; ``None`` = measure it
                on the GPU for every call (on a CPU everything goes into one batch).
            memory_padding: Fraction of the measured capacity actually used.
            steps_between_swaps: FIRE steps between convergence checks.
            show_progress: Show progress bars.
        """
        self.model = model
        self.max_memory_scaler = max_memory_scaler
        self.memory_padding = memory_padding
        self.steps_between_swaps = steps_between_swaps
        self.show_progress = show_progress
        self.capacities: list[float] = []
        if model.device.type == "cuda":
            # Batches of varying size fragment PyTorch's CUDA cache (up to a third of the GPU was seen
            # reserved but unusable); expandable segments avoid that.
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")  # noqa: SLF001
        self._measured: dict[bool, tuple[float, float, float]] = {}  # stress on/off: (min, max metric, capacity)

    def relax(self, structures: Sequence[Structure], *, fmax: float, max_steps: int) -> list[RelaxResult]:
        """Relax atoms and cell of every structure with FIRE on a Frechet cell filter, in batches.

        Args:
            structures: Structures to relax.
            fmax: FIRE stops when every force on atoms and cell is below this (eV/Å).
            max_steps: FIRE gives up after this many steps.

        Returns:
            One ``RelaxResult`` per structure, in input order.
        """
        if not structures:
            return []
        # Like ASE, structures that are already relaxed are not moved at all.
        starts = self.single_point(structures, compute_stress=True)
        todo = [
            i
            for i, (s, r) in enumerate(zip(structures, starts, strict=True))
            if not converged_before_relaxing(s, r, fmax)
        ]
        results = [_unmoved(structures[i], starts[i], fmax) for i in range(len(structures))]
        if todo:
            for i, relaxed in zip(
                todo, self._relax_batched([structures[i] for i in todo], fmax, max_steps), strict=True
            ):
                results[i] = relaxed
        return results

    def _relax_batched(self, structures: Sequence[Structure], fmax: float, max_steps: int) -> list[RelaxResult]:
        state = self._state(structures)

        def optimize(capacity: float) -> tuple[Any, Any]:
            batcher = ts.InFlightAutoBatcher(
                self.model, memory_scales_with=self.model.memory_scales_with, max_memory_scaler=capacity
            )
            final = ts.optimize(
                system=state,
                model=self.model,
                optimizer=(fire_init, ase_consistent_fire_step),
                convergence_fn=ase_convergence(fmax),
                max_steps=max_steps,
                steps_between_swaps=self.steps_between_swaps,
                autobatcher=batcher,
                pbar={"desc": "relax"} if self.show_progress else False,
                init_kwargs={"cell_filter": ts.CellFilter.frechet},
            )
            return final, batcher

        with _stress_enabled(self.model, enabled=True):
            final, batcher = self._batched(state, optimize)
        relaxed = ts.io.state_to_structures(final)
        energies = final.energy.detach().cpu().numpy()
        forces = _per_structure(final.forces, final)
        stresses = final.stress.detach().cpu().numpy()
        results = []
        for i, structure in enumerate(relaxed):
            max_force = float(np.linalg.norm(forces[i], axis=1).max())
            results.append(
                RelaxResult(
                    structure=structure,
                    energy=float(energies[i]),
                    forces=forces[i],
                    stress=stresses[i],
                    max_force=max_force,
                    converged=max_force <= fmax,  # atomic forces only, as ASESimulator and upstream
                    n_steps=batcher.iteration_count[i] * self.steps_between_swaps,
                )
            )
        return results

    def single_point(
        self, structures: Sequence[Structure | Atoms], *, compute_stress: bool = True
    ) -> list[SinglePointResult]:
        """Energy, forces and (optionally) stress of every structure, in batches.

        Args:
            structures: Structures to evaluate.
            compute_stress: Also compute the stress tensor (costs extra memory and time).

        Returns:
            One ``SinglePointResult`` per structure, in input order.
        """
        if not structures:
            return []
        state = self._state(structures)
        metric = calculate_memory_scalers(state, memory_scales_with=self.model.memory_scales_with)
        results: list[SinglePointResult] = [SinglePointResult.failed("not evaluated")] * state.n_systems
        with _stress_enabled(self.model, enabled=compute_stress):
            capacity = self._capacity(state, metric)
            batches = _pack(range(state.n_systems), metric, capacity)
            too_large: list[float] = []  # memory metric of structures that did not fit on their own
            progress = tqdm(total=state.n_systems, desc="single point", disable=not self.show_progress)
            while batches:
                batch = batches.pop(0)
                if len(batch) == 1 and any(np.isclose(metric[batch[0]], m, rtol=1e-9, atol=0) for m in too_large):
                    # Same number of atoms in the same volume (e.g. another displaced supercell of the same
                    # compound) as a structure that did not fit: not tried again.
                    results[batch[0]] = SinglePointResult.failed("GPU out of memory")
                    progress.update(1)
                    continue
                evaluated = self._evaluate(state[batch], compute_stress=compute_stress)
                if evaluated is None and len(batch) > 1:
                    # Lower the capacity below this batch for the rest of the call, so that running out of
                    # memory happens a few times per call rather than once per batch.
                    capacity = OUT_OF_MEMORY_BACKOFF * sum(metric[i] for i in batch)
                    self.capacities.append(capacity)
                    self._remember_lower_capacity(capacity)
                    logger.warning(
                        "GPU out of memory on a batch of %d structures; batch capacity lowered to %.4g",
                        len(batch),
                        capacity,
                    )
                    batches = _pack([i for b in (batch, *batches) for i in b], metric, capacity)
                    continue
                if evaluated is None:
                    logger.warning("A structure with %d atoms does not fit on the GPU", len(structures[batch[0]]))
                    too_large.append(float(metric[batch[0]]))
                    evaluated = [SinglePointResult.failed("GPU out of memory")]
                for index, result in zip(batch, evaluated, strict=True):
                    results[index] = result
                progress.update(len(batch))
            progress.close()
        return results

    def _evaluate(self, batch: Any, *, compute_stress: bool) -> list[SinglePointResult] | None:
        """One forward pass over a batch; ``None`` if the GPU ran out of memory."""
        try:
            out = self.model(batch)
        except RuntimeError as exc:
            if not _out_of_memory(exc):
                raise
            out = None
        if out is None:
            # Only now, outside the except block, is the failed forward pass (kept alive by the exception's
            # traceback) released, so its memory can be freed before the next attempt.
            _free_gpu_memory()
            return None
        energies = out["energy"].detach().cpu().numpy()
        forces = _per_structure(out["forces"], batch)
        stresses = out["stress"].detach().cpu().numpy() if compute_stress else [None] * batch.n_systems
        return [
            SinglePointResult(energy=float(energy), forces=force, stress=stress)
            for energy, force, stress in zip(energies, forces, stresses, strict=True)
        ]

    def _state(self, structures: Sequence[Structure | Atoms]) -> Any:
        return ts.io.atoms_to_state(
            [to_ase_atoms(structure) for structure in structures],
            device=self.model.device,
            dtype=self.model.dtype,
        )

    def _batched[T](self, state: Any, run: Callable[[float], T]) -> T:
        """Run ``run(capacity)`` on the GPU; after an out-of-memory error, free memory and retry.

        TorchSim itself cannot recover from running out of GPU memory in the middle of a run, and the
        usable memory can change while a job runs (fragmentation of PyTorch's cache, another process on a
        shared GPU). Each retry releases the cached memory and halves the capacity, down to the largest
        structure; at that size one more attempt is made before giving up.
        """
        metric = calculate_memory_scalers(state, memory_scales_with=self.model.memory_scales_with)
        # TorchSim recomputes the metric structure by structure, which can differ in the last digit.
        largest = max(metric) * (1 + 1e-6)
        capacity = max(self._capacity(state, metric), largest)  # every structure must fit into a batch
        last_try_at_smallest = False
        for attempt in range(MAX_OUT_OF_MEMORY_RETRIES + 1):
            try:
                return run(capacity)
            except RuntimeError as exc:
                if not _out_of_memory(exc) or attempt == MAX_OUT_OF_MEMORY_RETRIES or last_try_at_smallest:
                    raise
            _free_gpu_memory()  # outside the except block, so the failed run can be released
            last_try_at_smallest = capacity <= largest
            capacity = max(capacity / 2, largest)
            self.capacities.append(capacity)
            logger.warning("GPU out of memory; retrying with batch capacity %.4g", capacity)
        raise AssertionError("unreachable")  # pragma: no cover

    def _capacity(self, state: Any, metric: Sequence[float]) -> float:
        """Batch capacity for the structures of one call, in TorchSim's memory metric.

        Memory per unit of the metric differs between many tiny cells and a few large supercells, so the
        capacity is measured on the smallest and the largest structure of a call (TorchSim probes with
        growing copies of each until the GPU runs out of memory and backs off two steps). A later call
        whose structures lie within the range already measured (with the same stress setting) reuses the
        capacity (also when its smallest structure is down to half the smallest one measured); a call with
        a larger structure or much smaller ones is measured again. A capacity lowered after running out
        of memory is remembered.

        The capacity can be smaller than the largest structure: when that structure barely fits on the
        GPU on its own, TorchSim reports room for one copy of it. Single points then evaluate structures
        larger than the capacity one at a time; relaxations raise the capacity to the largest structure.
        """
        if self.max_memory_scaler is not None:
            capacity = self.max_memory_scaler
        elif self.model.device.type != "cuda":
            capacity = float(sum(metric)) * (1 + 1e-6) + 1.0  # no GPU memory to measure: one batch
        else:
            low, high, stress = min(metric), max(metric), self.model.compute_stress
            known = self._measured.get(stress)
            if known is not None and known[0] / 2 <= low and high <= known[1]:
                capacity = known[2]
            else:
                capacity = estimate_max_memory_scaler(
                    state,
                    self.model,
                    list(metric),
                    max_atoms=MAX_PROBE_ATOMS,
                    oom_error_message=OUT_OF_MEMORY_MESSAGES,
                )
                capacity *= self.memory_padding
                if known is not None:  # the range now covers both calls: keep the smaller capacity
                    low, high, capacity = min(low, known[0]), max(high, known[1]), min(capacity, known[2])
                self._measured[stress] = (low, high, capacity)
        self.capacities.append(capacity)
        logger.info("TorchSim batch capacity: %.4g (%d structures)", capacity, state.n_systems)
        return capacity

    def _remember_lower_capacity(self, capacity: float) -> None:
        known = self._measured.get(self.model.compute_stress)
        if known is not None:
            self._measured[self.model.compute_stress] = (known[0], known[1], min(known[2], capacity))


def _pack(indices: Iterable[int], metric: Sequence[float], capacity: float) -> list[list[int]]:
    """Pack structures into batches whose summed memory metric stays within ``capacity``, largest first.

    A structure larger than the capacity gets a batch of its own.
    """
    bins = to_constant_volume_bins({i: float(metric[i]) for i in indices}, max_volume=capacity)
    return [sorted(batch) for batch in bins]


def _free_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _out_of_memory(exc: BaseException) -> bool:
    # Out-of-memory errors raised inside TorchScript models arrive as plain RuntimeErrors.
    return any(message in str(exc) for message in OUT_OF_MEMORY_MESSAGES)


def _unmoved(structure: Structure | Atoms, start: SinglePointResult, fmax: float) -> RelaxResult:
    if start.error is not None or start.forces is None:
        return RelaxResult.failed(start.error or "single point failed")
    max_force = float(np.linalg.norm(start.forces, axis=1).max())
    return RelaxResult(
        structure=to_pmg_structure(structure),
        energy=start.energy,
        forces=start.forces,
        stress=start.stress,
        max_force=max_force,
        converged=max_force <= fmax,
        n_steps=0,
    )


def _per_structure(per_atom: torch.Tensor, state: Any) -> list[np.ndarray]:
    counts = state.n_atoms_per_system.tolist()
    return [block.detach().cpu().numpy() for block in torch.split(per_atom, counts)]


@contextmanager
def _stress_enabled(model: ModelInterface, *, enabled: bool) -> Iterator[None]:
    """Temporarily switch the stress calculation of a TorchSim model on or off.

    TorchSim models expose ``compute_stress`` read-only; the flag they read is ``_compute_stress``.
    Force-only single points (phonons, softening) skip the extra work of the stress.

    Args:
        model: TorchSim model.
        enabled: Whether stress should be computed inside the block.

    Yields:
        Nothing.
    """
    previous = model._compute_stress  # noqa: SLF001
    model._compute_stress = enabled  # noqa: SLF001
    try:
        yield
    finally:
        model._compute_stress = previous  # noqa: SLF001
