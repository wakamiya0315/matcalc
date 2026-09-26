"""Batched relaxations and single points on a GPU with TorchSim.

``ASESimulator`` sends one small cell at a time to the GPU, which leaves it mostly idle.
``TorchSimSimulator`` evaluates many structures in each forward pass instead: the structures of a batch
are concatenated into one graph, and batches are sized to fill the GPU memory.

- Relaxations use in-flight batching: when a structure has relaxed it leaves the batch and the next one
  takes its place. The optimizer is TorchSim's ASE-flavoured FIRE on a Frechet cell filter, the same
  algorithm as ``ASESimulator`` (ASE's FIRE on a ``FrechetCellFilter``).
- Single points are packed into batches by size.
- The batch capacity comes from TorchSim's memory probe (how many copies of the smallest and of the largest
  structure fit on the GPU), turned into an upper bound on the memory of every structure of a call.

TorchSim needs the MLIP as a TorchSim model (``torch_sim.models.interface.ModelInterface``), provided by
the user.
"""

from __future__ import annotations

import gc
import itertools
import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch_sim as ts
import torch_sim.math as tsm
from torch_sim.autobatching import calculate_memory_scalers, determine_max_batch_size, to_constant_volume_bins
from torch_sim.constraints import FixSymmetry
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


class BatchedFixSymmetry(FixSymmetry):
    """TorchSim's ``FixSymmetry`` constraint with all structures of a batch symmetrized at once.

    TorchSim symmetrizes forces, displacements, stress and cell steps structure by structure in Python,
    which costs about a third of a relaxation step for a batch of a hundred small cells. This subclass does
    the same arithmetic for the whole batch in a few tensor operations: every (symmetry operation, atom)
    pair of every structure is gathered, rotated and scatter-added at once. Results agree with
    ``FixSymmetry`` to rounding.
    """

    def _batch_indices(self, state: Any) -> dict[str, torch.Tensor]:
        """Index tensors of all (operation, atom) pairs, built once for a batch layout."""
        cached = getattr(self, "_indices", None)
        if cached is not None and cached["n_atoms"] == state.n_atoms and cached["device"] == state.device:
            return cached
        device = state.device
        counts = state.n_atoms_per_system.tolist()  # one transfer from the GPU
        offsets = [sum(counts[:i]) for i in range(len(counts))]
        rotations, op_structure, sources, targets, pair_ops, n_ops = [], [], [], [], [], []
        ops_of_atom = torch.ones(state.n_atoms, device=device, dtype=torch.long)
        constrained = torch.zeros(state.n_atoms, device=device, dtype=torch.bool)
        ops_so_far = 0
        for ci, si in enumerate(self.system_idx.tolist()):
            symm_map = self.symm_maps[ci].to(device)  # (n_ops, n_atoms): image of each atom under each op
            k, n = symm_map.shape
            offset = offsets[si]
            rotations.append(self.rotations[ci].to(device))
            op_structure.append(torch.full((k,), ci, device=device))
            sources.append(offset + torch.arange(n, device=device).repeat(k))
            targets.append(offset + symm_map.reshape(-1))
            pair_ops.append(ops_so_far + torch.arange(k, device=device).repeat_interleave(n))
            ops_of_atom[offset : offset + n] = k
            constrained[offset : offset + n] = True
            n_ops.append(k)
            ops_so_far += k
        systems = self.system_idx.to(device)
        n_ops_t = torch.tensor(n_ops, device=device)
        self._indices = {
            "n_atoms": state.n_atoms,
            "device": device,
            "systems": systems,
            "rotations": torch.cat(rotations),
            "op_structure": torch.cat(op_structure),
            "sources": torch.cat(sources),
            "targets": torch.cat(targets),
            "pair_ops": torch.cat(pair_ops),
            "n_ops": n_ops_t,
            "ops_of_atom": ops_of_atom,
            "constrained": constrained,
        }
        return self._indices

    def _symmetrize_rank1(self, state: Any, vectors: torch.Tensor) -> None:
        """Symmetrize per-atom vectors (forces, displacements) of every constrained structure in place."""
        index = self._batch_indices(state)
        lattice = state.row_vector_cell[state.system_idx]  # (n_atoms, 3, 3), rows = lattice vectors
        scaled = torch.einsum("ai,aij->aj", vectors, torch.linalg.inv(lattice))
        rotations = index["rotations"].to(vectors.dtype)[index["pair_ops"]]
        rotated = torch.einsum("pj,pkj->pk", scaled[index["sources"]], rotations)  # scaled @ R^T
        accumulated = torch.zeros_like(vectors).index_add_(0, index["targets"], rotated)
        averaged = accumulated / index["ops_of_atom"].to(vectors.dtype).unsqueeze(-1)
        symmetric = torch.einsum("ai,aij->aj", averaged, lattice)
        mask = index["constrained"]
        vectors[mask] = symmetric[mask]

    def _symmetrize_rank2(self, lattice: torch.Tensor, tensors: torch.Tensor, index: dict) -> torch.Tensor:
        """Symmetrize one rank-2 tensor per constrained structure (``lattice``: their cells as rows)."""
        rotations = index["rotations"].to(tensors.dtype)
        scaled = lattice @ tensors @ lattice.mT
        per_op = rotations.mT @ scaled[index["op_structure"]] @ rotations  # R^T S R for every operation
        summed = torch.zeros_like(scaled).index_add_(0, index["op_structure"], per_op)
        averaged = summed / index["n_ops"].to(tensors.dtype)[:, None, None]
        inverse = torch.linalg.inv(lattice)
        return inverse @ averaged @ inverse.mT

    def adjust_stress(self, state: Any, stress: torch.Tensor) -> None:
        """Symmetrize the stress of every constrained structure in place."""
        index = self._batch_indices(state)
        systems = index["systems"]
        stress[systems] = self._symmetrize_rank2(state.row_vector_cell[systems], stress[systems], index)

    def adjust_cell(self, state: Any, cell: torch.Tensor, max_delta_component: float = 0.25) -> None:
        """Symmetrize the step of every constrained cell in place, as ``FixSymmetry.adjust_cell``."""
        if not self.do_adjust_cell:
            return
        index = self._batch_indices(state)
        systems = index["systems"]
        identity = torch.eye(3, device=state.device, dtype=state.dtype)
        current = state.row_vector_cell[systems]
        delta = torch.linalg.solve(current, cell[systems].mT) - identity
        largest = delta.abs().amax(dim=(1, 2))
        if not bool(torch.isfinite(largest).all()):
            raise RuntimeError("FixSymmetry: deformation gradient is not finite; a cell may be singular.")
        # Large steps are shrunk before symmetrization, like TorchSim and ASE do.
        scale = torch.where(largest > max_delta_component, max_delta_component / largest, torch.ones_like(largest))
        delta = delta * scale[:, None, None]
        cell[systems] = (current @ (self._symmetrize_rank2(current, delta, index) + identity)).mT


class TorchSimSimulator:
    """Evaluate many structures per GPU forward pass with TorchSim.

    Attributes:
        model: TorchSim model of the MLIP.
        max_memory_scaler: Capacity of one batch in TorchSim's memory metric (sum over the batch of
            number of atoms x number density). ``None`` = derive it for every call from memory probes on
            the GPU (see ``_capacity``).
        memory_padding: Fraction of the measured memory actually used (TorchSim cannot recover from
            running out of GPU memory in the middle of a run).
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
        self._measures_memory = model.device.type == "cuda"
        # Per stress setting (on/off): probes of the smallest and the largest structure measured so far, and
        # the fraction of the probed memory batches may use (lowered after running out of memory).
        self._probes: dict[bool, tuple[MemoryProbe, MemoryProbe]] = {}
        self._budget: dict[bool, float] = {}

    def relax(
        self,
        structures: Sequence[Structure],
        *,
        fmax: float,
        max_steps: int,
        fix_symmetry: bool = False,
        symprec: float = 0.01,
    ) -> list[RelaxResult]:
        """Relax atoms and cell of every structure with FIRE on a Frechet cell filter, in batches.

        Args:
            structures: Structures to relax.
            fmax: FIRE stops when every force on atoms and cell is below this (eV/Å).
            max_steps: FIRE gives up after this many steps.
            fix_symmetry: Keep the space group of each structure with TorchSim's ``FixSymmetry``
                constraint (the counterpart of ASE's; needs ``moyopy``), applied to the whole batch at once
                (``BatchedFixSymmetry``).
            symprec: Symmetry tolerance used to find the space group (Å).

        Returns:
            One ``RelaxResult`` per structure, in input order.
        """
        if not structures:
            return []
        if fix_symmetry:
            # ASE's FixSymmetry first snaps the structure onto its symmetry; the same function is used here.
            structures = [_refined(structure, symprec) for structure in structures]
        # Like ASE, structures that are already relaxed are not moved at all.
        starts = self.single_point(structures, compute_stress=True)
        if fix_symmetry:
            starts = self._symmetrized(structures, starts, symprec)
        todo = [
            i
            for i, (s, r) in enumerate(zip(structures, starts, strict=True))
            if not converged_before_relaxing(s, r, fmax)
        ]
        results = [_unmoved(structures[i], starts[i], fmax) for i in range(len(structures))]
        if todo:
            relaxed = self._relax_batched(
                [structures[i] for i in todo], fmax, max_steps, fix_symmetry=fix_symmetry, symprec=symprec
            )
            for i, result in zip(todo, relaxed, strict=True):
                results[i] = result
        return results

    def _symmetrized(
        self, structures: Sequence[Structure], results: list[SinglePointResult], symprec: float
    ) -> list[SinglePointResult]:
        """Forces and stress symmetrized like the constraint does, for the test before the first step."""
        state = self._state(structures)
        constraint = BatchedFixSymmetry.from_state(state, symprec=symprec, refine_symmetry_state=False)
        good = [r.error is None and r.forces is not None and r.stress is not None for r in results]
        forces = torch.cat(
            [
                torch.as_tensor(r.forces if ok else np.zeros((len(s), 3)), dtype=state.dtype, device=state.device)
                for s, r, ok in zip(structures, results, good, strict=True)
            ]
        )
        stress = torch.stack(
            [
                torch.as_tensor(r.stress if ok else np.zeros((3, 3)), dtype=state.dtype, device=state.device)
                for r, ok in zip(results, good, strict=True)
            ]
        )
        constraint.adjust_forces(state, forces)
        constraint.adjust_stress(state, stress)
        per_structure = _per_structure(forces, state)
        return [
            replace(r, forces=per_structure[i], stress=stress[i].detach().cpu().numpy()) if good[i] else r
            for i, r in enumerate(results)
        ]

    def _relax_batched(
        self, structures: Sequence[Structure], fmax: float, max_steps: int, *, fix_symmetry: bool, symprec: float
    ) -> list[RelaxResult]:
        state = self._state(structures)
        if fix_symmetry:
            state.constraints = [BatchedFixSymmetry.from_state(state, symprec=symprec, refine_symmetry_state=False)]

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
        stopped = ase_convergence(fmax)(final).detach().cpu().numpy()
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
                    optimizer_converged=bool(stopped[i]),
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
            capacity, shares = self._capacity(state, metric)
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
                    if shares is not None:  # later calls: batches of at most this share of the probed memory
                        budget = self._budget.get(self.model.compute_stress, self.memory_padding)
                        lowered = OUT_OF_MEMORY_BACKOFF * float(shares[batch].sum())
                        self._budget[self.model.compute_stress] = min(budget, lowered)
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
        capacity = max(self._capacity(state, metric)[0], largest)  # every structure must fit into a batch
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

    def _capacity(self, state: Any, metric: Sequence[float]) -> tuple[float, np.ndarray | None]:
        """Batch capacity for the structures of one call, in TorchSim's memory metric.

        TorchSim measures how many copies of the smallest and of the largest structure (in its memory
        metric) fit on the GPU, backs off two steps of its search, and uses the smaller of the two
        capacities. That smaller one is set by the smallest structure, whose memory is mostly a fixed
        cost per structure and per atom that the metric ignores: a single one-atom cell with a large
        volume among the Equilibrium benchmark's elemental references cut the batches of all the other
        structures to a tenth of what fits. Here the same two probes give an upper bound on the memory of
        every structure of the call instead (``memory_shares``), and the capacity is the largest metric
        sum that no batch drawn from these structures can exceed the memory with (``batch_capacity``).

        The probes are cached per stress setting and measured again when a call has a structure larger
        than the largest one probed or smaller than half the smallest one. After running out of memory,
        later calls use batches of a smaller share of the probed memory.

        Returns:
            The capacity, and the memory share of every structure (``None`` if the capacity is not
            derived from probes).
        """
        shares = None
        if self.max_memory_scaler is not None:
            capacity = self.max_memory_scaler
        elif not self._measures_memory:
            capacity = float(sum(metric)) * (1 + 1e-6) + 1.0  # no GPU memory to measure: one batch
        else:
            metric = np.asarray(metric, dtype=float)
            n_atoms = state.n_atoms_per_system.detach().cpu().numpy().astype(float)
            shares = memory_shares(self._memory_probes(state, metric), n_atoms, metric)
            budget = self._budget.get(self.model.compute_stress, self.memory_padding)
            capacity = batch_capacity(shares, metric, budget)
        self.capacities.append(capacity)
        logger.info("TorchSim batch capacity: %.4g (%d structures)", capacity, state.n_systems)
        return capacity, shares

    def _memory_probes(self, state: Any, metric: np.ndarray) -> tuple[MemoryProbe, MemoryProbe]:
        """Probes of the smallest and the largest structure, from the cache or measured now."""
        stress = self.model.compute_stress
        cached = self._probes.get(stress)
        probed: dict[int, MemoryProbe] = {}

        def probe(index: int) -> MemoryProbe:
            if index not in probed:  # one structure can be both the smallest and the largest
                probed[index] = self._probe(state, index, metric)
            return probed[index]

        smallest, largest = int(np.argmin(metric)), int(np.argmax(metric))
        small = cached[0] if cached is not None and metric[smallest] >= cached[0].metric / 2 else probe(smallest)
        large = cached[1] if cached is not None and metric[largest] <= cached[1].metric else probe(largest)
        self._probes[stress] = (small, large)
        return small, large

    def _probe(self, state: Any, index: int, metric: np.ndarray) -> MemoryProbe:
        """How many copies of one structure fit on the GPU (TorchSim's search, which backs off two steps)."""
        copies = determine_max_batch_size(
            state[index], self.model, max_atoms=MAX_PROBE_ATOMS, oom_error_message=OUT_OF_MEMORY_MESSAGES
        )
        return MemoryProbe(n_atoms=int(state.n_atoms_per_system[index]), metric=float(metric[index]), copies=copies)


@dataclass(frozen=True)
class MemoryProbe:
    """Result of TorchSim's memory probe for one structure: ``copies`` copies of it fit into one batch.

    Attributes:
        n_atoms: Number of atoms of the structure.
        metric: Its value of the memory metric.
        copies: Number of copies that fit on the GPU, with TorchSim's safety margin.
    """

    n_atoms: int
    metric: float
    copies: int


def memory_shares(probes: tuple[MemoryProbe, MemoryProbe], n_atoms: np.ndarray, metric: np.ndarray) -> np.ndarray:
    """Upper bound on the share of the probed GPU memory that each structure needs in a batch.

    One copy of a probed structure needs ``1 / copies`` of the memory. The memory of a structure is taken to
    grow linearly, with non-negative costs, with three counts: structures (1), atoms and memory metric
    (for most MLIPs: node features per atom, messages per neighbour pair). A structure then needs no more
    than ``x`` copies of the small probe and ``y`` of the large one whenever that combination is at least
    as large in all three counts::

        x + y >= 1,    x n_small + y n_large >= n,    x m_small + y m_large >= m,    x, y >= 0

    and the bound is the smallest ``x / copies_small + y / copies_large`` over these combinations (a
    linear program in two variables, solved at the corners of its feasible region).

    Args:
        probes: Probes of a small and a large structure (may be the same).
        n_atoms: Number of atoms of every structure.
        metric: Memory metric of every structure.

    Returns:
        The share of every structure.
    """
    small, large = probes
    constraints = [
        (1.0, 1.0, np.ones_like(metric)),
        (float(small.n_atoms), float(large.n_atoms), n_atoms),
        (small.metric, large.metric, metric),
    ]  # each: coefficient of x, of y, and the count of the structure
    cost_x, cost_y = 1.0 / small.copies, 1.0 / large.copies
    corners = [
        cost_x * np.max([count / a for a, _, count in constraints], axis=0),  # only the small probe (y = 0)
        cost_y * np.max([count / b for _, b, count in constraints], axis=0),  # only the large probe (x = 0)
    ]
    for (a1, b1, c1), (a2, b2, c2) in itertools.combinations(constraints, 2):  # two constraints tight
        det = a1 * b2 - a2 * b1
        if abs(det) <= 1e-12 * max(abs(a1 * b2), abs(a2 * b1)):  # parallel: no corner
            continue
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        feasible = (x >= 0) & (y >= 0)
        for a, b, count in constraints:
            feasible &= a * x + b * y >= count * (1 - 1e-9)
        corners.append(np.where(feasible, cost_x * x + cost_y * y, np.inf))
    return np.min(corners, axis=0)


def batch_capacity(shares: np.ndarray, metric: np.ndarray, budget: float) -> float:
    """Largest metric sum below which every batch of these structures stays within a memory budget.

    Batches are limited by the sum of their memory metric; the worst batch for a given sum is made of the
    structures with the largest memory share per unit of metric. Taking them in that order (with a
    fraction of the last one) until the shares reach the budget gives the capacity: no set of these
    structures with a smaller metric sum has larger shares.

    Args:
        shares: Memory share of every structure (from ``memory_shares``).
        metric: Memory metric of every structure.
        budget: Share of the probed memory a batch may use.

    Returns:
        The capacity in units of the metric; the metric sum of all structures (plus rounding) if they all
        fit into one batch.
    """
    order = np.argsort(-(shares / metric), kind="stable")
    filled = np.cumsum(shares[order])
    if filled[-1] <= budget:
        return float(metric.sum()) * (1 + 1e-6)  # TorchSim recomputes the metric, to the last digit
    k = int(np.searchsorted(filled, budget, side="right"))  # the first structure that no longer fits whole
    before = float(filled[k - 1]) if k else 0.0
    return float(metric[order[:k]].sum()) + (budget - before) / float(shares[order[k]]) * float(metric[order[k]])


def _refined(structure: Structure | Atoms, symprec: float) -> Atoms:
    """A copy of the structure with its symmetry made exact, as ASE's ``FixSymmetry`` does on creation."""
    from ase.spacegroup.symmetrize import refine_symmetry

    atoms = to_ase_atoms(structure).copy()
    refine_symmetry(atoms, symprec)
    return atoms


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
        optimizer_converged=True,  # kept only for the structures that pass the test before the first step
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
