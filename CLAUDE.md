# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

A fork of materialyzeai/matcalc reduced to the four benchmarks (Equilibrium, Elasticity, Phonon,
Softening). `main` mirrors upstream and is never committed to; work happens on feature branches
(`refactor/benchmark-core`, then `feature/torchsim` on top of it). Numbers must stay equal to upstream
unless a change is listed under "Changed on purpose" in `README.md`.

## Common commands

```bash
pip install -e ".[mace,torchsim,benchmark,test]"
pytest tests -m "not mace"          # fast, offline, CPU (EMT potential on tiny datasets)
pytest tests -m mace                # real MACE model and datasets (GPU node, network)
ruff check src tests && ruff format --check src tests
mypy -p matcalc
matcalc-bench --benchmark elasticity --model MACE-MatPES-PBE-0 --n-samples 5 --out results/
matcalc-bench --backend torchsim --workers 4 --out results/     # batched GPU run of all four benchmarks
```

Heavy runs and GPU tests are done on TSUBAME4 (iqrsh for tests up to ~15 min, `gpu_h` jobs for timing
and full runs), from a checkout selected with `PYTHONPATH=<checkout>/src`, never inside the repository.

## Architecture

- `benchmarks/_common.py` — `Benchmark` base class: dataset loading and seeded subsampling, chunked
  `run()` with a JSON checkpoint (resume skips finished material ids), the result table
  (`<quantity>_DFT`, `<quantity>_<model>`, `status_<model>`), `summarize()` and per-stage timings
  (`with self.stage(name):`).
- `benchmarks/{equilibrium,elasticity,phonon,softening}.py` — one benchmark each. `read_entries()` parses
  the dataset; `evaluate(materials, simulator)` is the recipe: stages over all materials of a chunk.
- `properties/` — physics as pure functions with no model code (elastic fit, phonopy, formation energy,
  softening scale, fingerprints). Keep them independent of the simulator.
- `simulation/` — the simulator interface (`relax`, `single_point`, result dataclasses in `base.py`) and
  `ASESimulator` (FIRE + FrechetCellFilter, one structure at a time; the reference implementation).
  `as_simulator()` accepts a simulator, an ASE calculator, a TorchSim model, or a MACE model name.
- `simulation/torchsim.py` — `TorchSimSimulator`: batched `relax` (in-flight FIRE + Frechet filter) and
  `single_point` (binned batches). It must stay step-for-step identical to `ASESimulator`: the FIRE step
  wrapper (`ase_consistent_fire_step`), `ase_convergence` and `converged_before_relaxing` exist for that
  and are checked by `tests/test_simulation_torchsim.py` (same energies to 1e-9 eV and same step counts
  as ASE). Batch capacity is measured per call and cached per metric range; out-of-memory errors split
  batches instead of failing. `GrowingNeighborList` avoids nvalchemiops' fixed neighbour cap.
- CPU post-processing (phonopy, fingerprints) runs through `parallel_map` (spawn processes); scripts that
  use `workers > 1` need an `if __name__ == "__main__":` guard.
- `datasets.py` (HF download, `sample_subset`), `models.py` (`load_mace`), `cli.py` (`matcalc-bench`).

## Conventions

- Readability for chemists: names state the physics, docstrings state units and meaning, settings are
  explicit keyword arguments with upstream defaults. Comments, docstrings and docs are in English.
- Failures of single materials never stop a run: simulators return results with `error`, benchmarks write
  the reason into `status` and NaN into the quantities.
- All modules start with `from __future__ import annotations`; ruff runs with `select = ["ALL"]`
  (see `pyproject.toml` for ignores). Google-style docstrings, type hints everywhere.
- Tests build tiny datasets from EMT-describable crystals in `tests/helpers.py`/`tests/conftest.py`;
  anything needing MACE or the network is marked `mace`/`network`.
