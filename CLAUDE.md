# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

A fork of materialyzeai/matcalc reduced to the four benchmarks (Equilibrium, Elasticity, Phonon,
Softening). `main` mirrors upstream and is never committed to; work happens on feature branches
(`refactor/benchmark-core`, then `feature/torchsim` on top of it). Numbers must stay equal to upstream
unless a change is listed under "Changed on purpose" in `README.md`.

## Common commands

```bash
pip install -e ".[mace,benchmark,test]"
pytest tests -m "not mace"          # fast, offline, CPU (EMT potential on tiny datasets)
pytest tests -m mace                # real MACE model and datasets (GPU node, network)
ruff check src tests && ruff format --check src tests
mypy -p matcalc
matcalc-bench --benchmark elasticity --model MACE-MatPES-PBE-0 --n-samples 5 --out results/
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
  `as_simulator()` accepts a simulator, an ASE calculator, or a MACE model name.
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
