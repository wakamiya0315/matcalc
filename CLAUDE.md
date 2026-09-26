# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

A fork of materialyzeai/matcalc reduced to the four benchmarks (Equilibrium, Elasticity, Phonon,
Softening). `main` mirrors upstream and is never committed to; the work is on `feature/torchsim` (the
refactoring and the TorchSim simulator). Equilibrium, Elasticity and Softening
must give the same numbers as upstream unless a change is listed under "Changed on purpose" in
`README.md`; the Phonon benchmark follows the protocol of its DFT reference (Alexandria, Loew et al. 2025),
not upstream.

The library contains only the benchmarks and the generic simulators. It does not load or depend on any
MLIP: the user passes an ASE calculator or a TorchSim model. MACE (MACE-MatPES-PBE-0) is only the test
model of the validation runs (`validation/mace_models.py`). Speed-ups belong on the benchmark side and
must help any MLIP (no model-specific kernels or precision tricks).

## Common commands

```bash
pip install -e ".[torchsim,benchmark,test]"
pytest tests                        # fast, offline, CPU (EMT and Lennard-Jones on tiny datasets)
ruff check src tests && ruff format --check src tests
mypy -p matcalc
python scripts/build_phonon_dataset.py temp/alexandria src/matcalc/benchmarks/data/alexandria-pbe-phonon.json.gz
```

`temp/` (git-ignored) holds downloads such as Alexandria's phonopy files. Heavy runs and GPU tests are done
on TSUBAME4 (iqrsh for tests up to ~15 min, `gpu_h` jobs for timing and full runs), from a checkout
selected with `PYTHONPATH=<checkout>/src`, never inside the repository; `validation/run_one.py` runs one
benchmark with MACE.

## Architecture

- `benchmarks/_common.py` — `Benchmark` base class: dataset loading (Hugging Face name or local `Path`)
  and seeded subsampling, chunked `run()` with a JSON checkpoint (resume skips finished material ids), the
  result table (`<quantity>_DFT`, `<quantity>_<model>`, `status_<model>`), `summarize()` and per-stage
  timings (`with self.stage(name):`). `Material.settings` carries DFT settings a benchmark reuses.
- `benchmarks/{equilibrium,elasticity,phonon,softening}.py` — one benchmark each. `read_entries()` parses
  the dataset; `evaluate(materials, simulator)` is the recipe: stages over all materials of a chunk.
  `benchmarks/data/alexandria-pbe-phonon.json.gz` is the Phonon dataset (unit cells, supercell and
  primitive matrices, displacements, C_V and stability of the DFT calculations).
- `properties/` — physics as pure functions with no model code (elastic fit, phonopy, formation energy,
  softening scale, fingerprints). Keep them independent of the simulator.
- `simulation/` — the simulator interface (`relax`, `single_point`, result dataclasses in `base.py`) and
  `ASESimulator` (FIRE + FrechetCellFilter, optionally ASE's `FixSymmetry`, one structure at a time; the
  reference implementation). `as_simulator()` accepts a simulator, an ASE calculator or a TorchSim model.
- `simulation/torchsim.py` — `TorchSimSimulator`: batched `relax` (in-flight FIRE + Frechet filter,
  optionally TorchSim's `FixSymmetry`) and `single_point` (packed batches). It must stay step-for-step
  identical to `ASESimulator`: the FIRE step wrapper (`ase_consistent_fire_step`), `ase_convergence` and
  `converged_before_relaxing` exist for that and are checked by `tests/test_simulation_torchsim.py` (same
  energies and step counts as ASE, with and without the symmetry constraint). The batch capacity is
  derived from TorchSim's memory probes of the smallest and the largest structure (cached), which bound
  the memory of every structure of a call (`memory_shares`, `batch_capacity`); after running out of memory
  the capacity is lowered.
- CPU post-processing (phonopy, stress-strain fits, fingerprints) runs in a `worker_pool` of spawned
  processes (`pool_map`), overlapping with the GPU: single points are computed in parts
  (`split_into_parts`) and the CPU work of one part runs while the GPU computes the next. Scripts that use
  `workers > 1` need an `if __name__ == "__main__":` guard.
- `datasets.py` (HF download, `sample_subset`); `scripts/build_phonon_dataset.py`.

## Conventions

- Readability for chemists: names state the physics, docstrings state units and meaning, settings are
  explicit keyword arguments with the reference's defaults. Comments, docstrings and docs are in English.
- Failures of single materials never stop a run: simulators return results with `error`, benchmarks write
  the reason into `status` and NaN into the quantities.
- All modules start with `from __future__ import annotations`; ruff runs with `select = ["ALL"]`
  (see `pyproject.toml` for ignores). Google-style docstrings, type hints everywhere.
- Tests build tiny datasets from EMT-describable crystals in `tests/helpers.py`/`tests/conftest.py`;
  anything needing the network is marked `network`.
