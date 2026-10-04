# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this fork is

A fork of materialyzeai/matcalc reduced to the four benchmarks (Equilibrium, Elasticity, Phonon,
Softening), plus three Matbench Discovery tasks (Discovery: WBM stability and geometry; Kappa: κ_SRME; Diatomics: dimer curves)
and three molecular tasks as MLIPAudit computes them (Noncovalent: NCI Atlas interaction energies;
Conformers: Folmsbee–Hutchison conformer energies; Reactions: RDB7 barrier heights), plus GMTKN55 (WTMAD-2) and
Adsorption (ADS41 and the Surf13 systems, against experiment, in configurations fixed by the dataset).
`main` is this fork (the refactoring and the TorchSim simulator); changes reach it through
pull requests from feature branches, and CI (`.github/workflows/`) runs ruff, mypy and the CPU tests on
them. `upstream-main` mirrors upstream's `main` and is never committed to (update it with
`git fetch upstream && git push origin upstream/main:upstream-main`). Equilibrium, Elasticity and Softening
must give the same numbers as upstream unless a change is listed under "Changed on purpose" in
`README.md`; the Phonon benchmark follows the protocol of its DFT reference (Alexandria, Loew et al. 2025),
not upstream.

The library contains only the benchmarks and the generic simulators. It does not load or depend on any
MLIP: the user passes an ASE calculator or a TorchSim model. MACE (MACE-MatPES-PBE-0; MACE-MP-0 and
MACE-OFF23 where their published results validate a benchmark) is only the test model of the validation
runs (`validation/mace_models.py`). Speed-ups belong on the benchmark side and
must help any MLIP (no model-specific kernels or precision tricks).

## Common commands

```bash
pip install -e ".[torchsim,benchmark,test]"
pytest tests                        # fast, offline, CPU (EMT and Lennard-Jones on tiny datasets)
ruff check src tests && ruff format --check src tests
mypy -p matcalc
python scripts/build_phonon_dataset.py temp/alexandria src/matcalc/benchmarks/data/alexandria-pbe-phonon.json.gz
python scripts/build_kappa_dataset.py temp/phonondb-pbe temp/matbench-discovery/<structures>.extxyz \
    temp/matbench-discovery/<published kappas>.json.gz src/matcalc/benchmarks/data/phonondb-pbe-kappa.json.gz \
    --workers 4 --cache temp/kappa-cache   # about 45 min on 10 cores; the cache keeps finished compounds
python scripts/build_adsorption_dataset.py src/matcalc/benchmarks/data/adsorption.json --images temp/adsorption
    # downloads Shi et al.'s structures once (MD5-checked); --images draws every configuration for checking
```

`temp/` (git-ignored) holds downloads such as Alexandria's phonopy files. Heavy runs and GPU tests are done
on TSUBAME4 (iqrsh for tests up to ~15 min, `gpu_h` jobs for timing and full runs), from a checkout
selected with `PYTHONPATH=<checkout>/src`, never inside the repository; `validation/run_one.py` runs one
benchmark with MACE.

## Architecture

- `benchmarks/_common.py` — `Benchmark` base class: dataset loading (Hugging Face name or local `Path`)
  and seeded subsampling, chunked `run()` with a JSON checkpoint (resume skips finished material ids), the
  result table (`<quantity>_DFT`, or `<quantity>_ref` with `reference_label = "ref"` for the coupled-cluster
  references of the molecular benchmarks; `<quantity>_<model>`, `status_<model>`), `summarize()` and
  per-stage timings (`with self.stage(name):`). `Material.settings` carries DFT settings a benchmark reuses.
- `benchmarks/{equilibrium,elasticity,phonon,softening,discovery,kappa,diatomics,noncovalent,conformers,reactions,gmtkn55,adsorption}.py` — one benchmark each. `read_entries()` parses
  the dataset; `evaluate(materials, simulator)` is the recipe: stages over all materials of a chunk.
  `benchmarks/data/alexandria-pbe-phonon.json.gz` is the Phonon dataset (unit cells, supercell and
  primitive matrices, displacements, C_V and stability of the DFT calculations);
  `benchmarks/data/phonondb-pbe-kappa.json.gz` the Kappa dataset (PhononDB cells and the PBE conductivity
  recomputed with the phono3py the benchmark uses). Discovery reads Matbench Discovery's Figshare files
  (`datasets.download_figshare_file`, MD5-checked) and keeps its 257,000 structures as ASE `Atoms`; it can
  run in shards (`shard=(k, n)`, one job each) whose tables `DiscoveryBenchmark.merge_shards` joins.
  Noncovalent, Conformers and GMTKN55 read files pinned to a commit on GitHub, Reactions a Zenodo file of RDB7
  (`datasets.download_file`, MD5-checked); their molecules sit in a 50 Å periodic box (`structures.molecule_in_box`) with the total
  charge and spin multiplicity in `Atoms.info`, which `TorchSimSimulator` passes into the TorchSim state.
  Adsorption reads `benchmarks/data/adsorption.json` (crystals, slabs, molecules, adsorbed structures as offsets
  from an anchor site or surface atom, and the reactions as coefficients of these structures); the configurations
  are decided in `scripts/build_adsorption_dataset.py` (sources cited per structure), not searched at run time.
  It relaxes crystals (cell, symmetry kept), then slabs cut from them, then adsorbed slabs and molecules.
- `properties/` — physics as pure functions with no model code (elastic fit, phonopy, formation energy,
  softening scale, fingerprints). Keep them independent of the simulator.
- `simulation/` — the simulator interface (`relax`, `single_point`, result dataclasses in `base.py`) and
  `ASESimulator` (FIRE + FrechetCellFilter, optionally ASE's `FixSymmetry`, one structure at a time; the
  reference implementation). `relax(..., relax_cell=False)` runs FIRE on the atoms alone in a fixed cell and
  keeps ASE `FixAtoms` constraints (slabs); single points ignore constraints. `as_simulator()` accepts a
  simulator, an ASE calculator or a TorchSim model.
- `simulation/torchsim.py` — `TorchSimSimulator`: batched `relax` (in-flight FIRE + Frechet filter,
  optionally TorchSim's `FixSymmetry`) and `single_point` (packed batches). It must stay step-for-step
  identical to `ASESimulator`: the FIRE step wrapper (`ase_consistent_fire_step`), `ase_convergence` and
  `converged_before_relaxing` exist for that and are checked by `tests/test_simulation_torchsim.py` (same
  energies and step counts as ASE, with and without the symmetry constraint, and in a fixed cell with fixed atoms,
  which become TorchSim's `FixAtoms`; other ASE constraints are refused). The batch capacity is
  derived from TorchSim's memory probes of the smallest and the largest structure (cached), which bound
  the memory of every structure of a call (`memory_shares`, `batch_capacity`); after running out of memory
  the capacity is lowered.
- CPU post-processing (phonopy, stress-strain fits, fingerprints) runs in a `worker_pool` of spawned
  processes (`pool_map`), overlapping with the GPU: single points are computed in parts
  (`split_into_parts`) and the CPU work of one part runs while the GPU computes the next. Scripts that use
  `workers > 1` need an `if __name__ == "__main__":` guard.
- `datasets.py` (Hugging Face, Figshare and GitHub downloads, `sample_subset`); `structures.py`
  (pymatgen/ASE conversions, `molecule_in_box`); `surfaces.py` (bulk crystals, metal slabs from ASE's builders,
  oxide slabs from an oriented bulk cell with Tasker's non-polar termination, anchors and adsorbate placement);
  `scripts/build_phonon_dataset.py`, `scripts/build_kappa_dataset.py`, `scripts/build_adsorption_dataset.py`.

## Conventions

- Readability for chemists: names state the physics, docstrings state units and meaning, settings are
  explicit keyword arguments with the reference's defaults. Comments, docstrings and docs are in English.
- Failures of single materials never stop a run: simulators return results with `error`, benchmarks write
  the reason into `status` and NaN into the quantities.
- All modules start with `from __future__ import annotations`; ruff runs with `select = ["ALL"]`
  (see `pyproject.toml` for ignores). Google-style docstrings, type hints everywhere.
- Tests build tiny datasets from EMT-describable crystals in `tests/helpers.py`/`tests/conftest.py`;
  anything needing the network is marked `network`.
