# MatCalc benchmarks (fork)

This is a fork of [materialyzeai/matcalc](https://github.com/materialyzeai/matcalc) reduced to its
four benchmarks for machine-learning interatomic potentials (MLIPs):

| Benchmark | Compared with DFT (PBE) | Dataset |
|---|---|---|
| Equilibrium | formation energy `Eform` (eV/atom) and structural distance `d` after relaxation | 972 WBM compounds |
| Elasticity | bulk and shear moduli `K_vrh`, `G_vrh` (GPa) | 3,953 binary compounds (Materials Project) |
| Phonon | heat capacity `CV` at 300 K (J/(K·mol)) | 1,170 binary compounds (Alexandria) |
| Softening | slope of MLIP forces against DFT forces on high-energy configurations | 979 WBM materials, 9,308 frames |

The `main` branch is the unmodified upstream code. This branch keeps only what the benchmarks need and
gives the same numbers as upstream (see [Differences from upstream](#differences-from-upstream)).
What each benchmark computes, step by step and with units, is described in
[docs/benchmarks.md](docs/benchmarks.md).

## Install

```bash
pip install -e ".[mace,torchsim,benchmark]"
```

`mace` installs `mace-torch` for `matcalc.load_mace`; `torchsim` installs TorchSim (`torch-sim-atomistic`
≥ 0.6.2) for batched GPU runs; `benchmark` installs `matminer`, needed for the structural distance of the
Equilibrium benchmark. Any other MLIP can be used through its ASE calculator (or its TorchSim model).

## Run

From the command line (the datasets are downloaded from Hugging Face on first use):

```bash
matcalc-bench --benchmark equilibrium elasticity phonon softening \
    --model MACE-MatPES-PBE-0 --n-samples 20 --out results/
```

This writes, per benchmark, `<benchmark>_<model>.csv` (the result table), `<benchmark>_<model>.json.gz`
(all rows including structures; an interrupted run resumes from it) and `summary.json` (MAE against DFT
and wall time per stage).

Add `--backend torchsim` to evaluate many structures per GPU forward pass with TorchSim (see
[below](#batched-gpu-runs-with-torchsim)).

From Python:

```python
import matcalc

calculator = matcalc.load_mace("MACE-MatPES-PBE-0")  # or any ASE calculator
benchmark = matcalc.ElasticityBenchmark(n_samples=20, seed=42)
table = benchmark.run(calculator, "MACE", checkpoint_file="elasticity_MACE.json.gz")
print(benchmark.summarize(table, "MACE"))
```

## Batched GPU runs with TorchSim

`ASESimulator` evaluates one small cell per GPU call, which leaves a GPU mostly idle. `TorchSimSimulator`
runs the same two operations with [TorchSim](https://github.com/TorchSim/torch-sim), packing many
structures into each forward pass:

```python
import matcalc
from matcalc.simulation import TorchSimSimulator

model = matcalc.load_mace("MACE-MatPES-PBE-0", backend="torchsim")  # same checkpoint as the ASE calculator
simulator = TorchSimSimulator(model)
table = matcalc.PhononBenchmark(n_samples=20).run(simulator, "MACE")
```

- Relaxations use in-flight batching (a relaxed structure leaves the batch and the next one joins) with
  TorchSim's FIRE on a Frechet cell filter. Three details are adjusted so that every structure follows
  exactly the path of ASE's FIRE on a `FrechetCellFilter` whatever batch it is in: the first step of a
  structure joining a running batch, the order in which FIRE's mixing parameter is updated, and ASE's
  convergence test (transformed atomic forces plus cell forces, checked before the first step too).
- Single points are packed into batches by size; force-only single points (phonons, softening) skip the
  stress. A structure larger than the batch capacity is evaluated on its own, and one that does not fit
  on the GPU even alone gets no prediction (as with the ASE path).
- The batch capacity is measured on the GPU (TorchSim's memory probe). When a batch runs out of memory the
  capacity is lowered for the rest of the call; a relaxation is retried with half the capacity.
- `SplitSimulator(relaxer, evaluator)` relaxes with one simulator and computes the single points with
  another. `matcalc-bench --backend torchsim --fast-single-points` uses it to keep the relaxations in
  float64 and compute the single points in float32 with cuEquivariance kernels, which makes the phonon
  supercells 10–20× faster (accuracy in [docs/validation.md](docs/validation.md)).

Agreement with the ASE path and wall times are reported in [docs/validation.md](docs/validation.md).

Each result table has one row per material: the id, the formula, the DFT values (`<quantity>_DFT`), the
predictions (`<quantity>_<model>`), and `status_<model>`, which is `ok` or the reason a prediction is
missing (for example a relaxation that did not converge). The column names are those of upstream matcalc.

## How the code is organized

```
src/matcalc/
  benchmarks/   one file per benchmark; each reads as a recipe of stages
                (relax everything → build strained/displaced cells → evaluate them → fit)
  properties/   the physics as plain functions: elastic fit, phonons, formation energy,
                softening scale, structural fingerprints
  simulation/   simulators: what evaluates the MLIP. ASESimulator relaxes (FIRE + Frechet cell
                filter) and computes single points with any ASE calculator, one structure at a time;
                TorchSimSimulator does the same in batches on the GPU
  datasets.py   download and reproducible subsets
  models.py     load_mace()
  cli.py        matcalc-bench
```

A benchmark only asks its simulator for two operations, `relax(structures, fmax, max_steps)` and
`single_point(structures)`, always for all materials of a chunk at once.

## Differences from upstream

Removed: every calculator the benchmarks do not use (adsorption, EOS, grain boundaries, interfaces,
LAMMPS, MD, NEB, order, phonon3, QHA, surfaces), `ChainedCalc`, the multi-provider model registry, the old
CLI, example notebooks, the documentation site, and the classical-potential test files.

Changed on purpose, affecting the numbers:

- The Phonon supercells are at least 15 Å long along each lattice vector instead of 20 Å, which halves
  the work. C_V changes only for compounds that are dynamically unstable with the MLIP (see
  [docs/validation.md](docs/validation.md)); `PhononBenchmark(min_supercell_length=20)` or
  `matcalc-bench --min-supercell-length 20` reproduces upstream.

Changed on purpose (the numbers of a successful run are not affected):

- The Equilibrium benchmark displaces the atoms with a generator seeded by `seed`; upstream used an
  unseeded generator, so its results changed from run to run.
- The Phonon benchmark builds compact force constants and no eigenvectors, and keeps only the thermal
  properties instead of whole `Phonopy` objects (the heat capacity is identical; memory is much smaller).
- Checkpoints are JSON files with honest extensions and can resume any benchmark.
- Bug fixes: `PhononBenchmark` options raised `TypeError`; `SofteningBenchmark` could not be combined with
  other benchmarks and returned only `None` when given a model name; subsampling changed the global
  `random` state (the same subsets are still drawn); results of several models were joined on row
  position instead of the material id; importing matcalc queried the network; an Equilibrium run crashed
  at the end (`AttributeError` in the fingerprint step) as soon as one relaxation had not converged.

Kept on purpose, for comparability with published numbers:

- Convergence of a relaxation is judged on the atomic forces only (not on the residual cell stress).
- The Phonon benchmark uses the relaxed structure even if its relaxation did not converge, while the
  Elasticity and Equilibrium benchmarks give no prediction in that case.
- The phonon supercell repeats the cell `ceil(L / |a_i|)` times along each lattice vector, which leaves
  some rhombohedral cells with 1×1×1 supercells only a few Å thick.
- Equilibrium atoms are displaced by a random distance of at most 0.1 Å (pymatgen's `Structure.perturb`
  with `min_distance=0`), not by exactly 0.1 Å.

## Citing

Please cite matcalc (see [`citation.cff`](citation.cff)) and the papers of the datasets and models you use;
for the softening benchmark, B. Deng et al., npj Comput. Mater. 11, 9 (2025).
