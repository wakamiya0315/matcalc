# MatCalc benchmarks (fork)

This is a fork of [materialyzeai/matcalc](https://github.com/materialyzeai/matcalc) reduced to its
four benchmarks for machine-learning interatomic potentials (MLIPs), plus two tasks of
[Matbench Discovery](https://matbench-discovery.materialsproject.org) that need only relaxations and
single points as well:

| Benchmark | Compared with DFT (PBE) | Dataset |
|---|---|---|
| Equilibrium | formation energy `Eform` (eV/atom) and structural distance `d` after relaxation | 972 WBM compounds |
| Elasticity | bulk and shear moduli `K_vrh`, `G_vrh` (GPa) | 3,953 binary compounds (Materials Project) |
| Phonon | heat capacity `CV` at 300 K (J/(K·mol)) and dynamical stability | 1,170 binary compounds (Alexandria) |
| Softening | slope of MLIP forces against DFT forces on high-energy configurations | 979 WBM materials, 9,308 frames |
| Discovery | stability against the convex hull (F1, DAF, `e_above_hull` MAE) and relaxed geometry (RMSD, symmetry) | 256,963 WBM crystals (Matbench Discovery) |
| Kappa | lattice thermal conductivity at 300 K (κ_SRME, κ_SRE) | 103 PhononDB crystals (Matbench Discovery) |

The unmodified upstream code (materialyzeai/matcalc at `b04715d`, 2026-09-09) is kept on the branch
[`upstream-main`](https://github.com/wakamiya0315/matcalc/tree/upstream-main). Equilibrium,
Elasticity and Softening give the same numbers as upstream; the Phonon benchmark follows the protocol of
its DFT reference instead (see [Differences from upstream](#differences-from-upstream)). What each
benchmark computes, step by step and with units, is described in [docs/benchmarks.md](docs/benchmarks.md).

## Install

```bash
git clone https://github.com/wakamiya0315/matcalc.git && cd matcalc
pip install -e ".[torchsim,benchmark]"
```

or, without a clone, `pip install "matcalc[torchsim,benchmark] @ git+https://github.com/wakamiya0315/matcalc.git"`.
The package is still called `matcalc` and replaces upstream's in the same environment.

`torchsim` installs TorchSim (`torch-sim-atomistic` ≥ 0.6.2, and `moyopy` for its symmetry constraint) for
batched GPU runs; `benchmark` installs `matminer` and `moyopy`, needed for the structural distance of the
Equilibrium benchmark and the space groups of the Discovery benchmark; `kappa` installs phono3py for the
Kappa benchmark. The MLIP itself is not part of matcalc: install it separately and pass its ASE calculator (or
its TorchSim model).

## Run

```python
import matcalc

calculator = ...  # the ASE calculator of the MLIP to benchmark
benchmark = matcalc.ElasticityBenchmark(n_samples=20, seed=42)
table = benchmark.run(calculator, "my-mlip", checkpoint_file="elasticity_my-mlip.json.gz")
print(benchmark.summarize(table, "my-mlip"))
```

The Equilibrium, Elasticity and Softening datasets are downloaded from Hugging Face on first use; the
Phonon dataset is part of the package. `run` returns the result table; the checkpoint file keeps all
finished rows (including structures), so an interrupted run resumes from it. `matcalc.run_benchmarks`
runs several benchmarks for several models and writes the tables.

## Batched GPU runs with TorchSim

`ASESimulator` evaluates one small cell per GPU call, which leaves a GPU mostly idle. `TorchSimSimulator`
runs the same two operations with [TorchSim](https://github.com/TorchSim/torch-sim), packing many
structures into each forward pass:

```python
import matcalc
from matcalc.simulation import TorchSimSimulator

model = ...  # the TorchSim model of the MLIP (torch_sim.models.interface.ModelInterface)
simulator = TorchSimSimulator(model)
table = matcalc.PhononBenchmark(workers=4).run(simulator, "my-mlip")
```

- Relaxations use in-flight batching (a relaxed structure leaves the batch and the next one joins) with
  TorchSim's FIRE on a Frechet cell filter. Three details are adjusted so that every structure follows
  exactly the path of ASE's FIRE on a `FrechetCellFilter` whatever batch it is in: the first step of a
  structure joining a running batch, the order in which FIRE's mixing parameter is updated, and ASE's
  convergence test (transformed atomic forces plus cell forces, checked before the first step too).
  With `fix_symmetry=True` (Phonon and Kappa benchmarks) TorchSim's `FixSymmetry` constraint plays the
  role of ASE's.
- Single points are packed into batches by size; force-only single points (phonons, softening) skip the
  stress. A structure larger than the batch capacity is evaluated on its own, and one that does not fit
  on the GPU even alone gets no prediction (as with the ASE path).
- The batch capacity comes from TorchSim's memory probe on the GPU (how many copies of the smallest and of
  the largest structure fit). TorchSim would size every batch for copies of the smallest structure, whose
  memory is mostly a fixed cost that its memory metric ignores; instead the two probes bound the memory of
  each structure of a call, so that a few tiny cells no longer shrink the batches of all the others. When a
  batch runs out of memory the capacity is lowered; a relaxation is retried with half the capacity.
- The CPU work (phonopy and phono3py, structural fingerprints, structure matching and symmetry) runs in
  `workers` parallel processes, as far as possible while the GPU computes the next structures.

Agreement with the ASE path and wall times are reported in [docs/validation.md](docs/validation.md).

Each result table has one row per material: the id, the formula, the DFT values (`<quantity>_DFT`), the
predictions (`<quantity>_<model>`), and `status_<model>`, which is `ok` or the reason a prediction is
missing (for example a relaxation that did not converge). The column names are those of upstream matcalc.

## Tests

```bash
pip install -e ".[torchsim,benchmark,test]"
pytest tests
```

The tests run on the CPU in a few minutes, with the EMT and Lennard-Jones potentials on tiny datasets
(TorchSimSimulator is checked against ASESimulator on the same potential). GitHub Actions runs them, and
ruff and mypy, on every push and pull request to `main`. GPU runs with a real MLIP are described in
[docs/validation.md](docs/validation.md).

## How the code is organized

```
src/matcalc/
  benchmarks/   one file per benchmark; each reads as a recipe of stages
                (relax everything → build strained/displaced cells → evaluate them → fit);
                data/ holds the Phonon and Kappa datasets (settings and results of the DFT
                phonon calculations)
  properties/   the physics as plain functions: elastic fit, phonons, formation energy,
                softening scale, structural fingerprints, stability and geometry metrics,
                thermal conductivity
  simulation/   simulators: what evaluates the MLIP. ASESimulator relaxes (FIRE + Frechet cell
                filter) and computes single points with any ASE calculator, one structure at a time;
                TorchSimSimulator does the same in batches on the GPU
  datasets.py   download and reproducible subsets
scripts/        build_phonon_dataset.py: the Phonon dataset from Alexandria's phonopy files;
                build_kappa_dataset.py: the Kappa dataset from PhononDB's phono3py force sets
validation/     the runs behind docs/validation.md (with MACE as the test model)
```

A benchmark only asks its simulator for two operations, `relax(structures, fmax, max_steps)` and
`single_point(structures)`, always for all materials of a chunk at once.

## Differences from upstream

Added: the Discovery and Kappa benchmarks, Matbench Discovery's discovery (with geometry optimization)
and κ_SRME tasks written in the same form as the others (see [docs/benchmarks.md](docs/benchmarks.md)).
Their data come from Matbench Discovery's Figshare (downloaded on first use) and, for the Kappa
reference, from PhononDB's PBE force sets (packaged); both are CC BY 4.0.

Removed: every calculator the benchmarks do not use (adsorption, EOS, grain boundaries, interfaces,
LAMMPS, MD, NEB, order, phonon3, QHA, surfaces), `ChainedCalc`, the multi-provider model registry, the
CLI, example notebooks, the documentation site, and the classical-potential test files.

Changed on purpose, affecting the numbers — the Phonon benchmark repeats the phonon calculation of its
DFT reference (A. Loew et al., npj Comput. Mater. 2025; Alexandria, CC BY 4.0) with the MLIP:

- it starts from the PBE unit cell of that calculation and relaxes it keeping its space group, to
  0.005 eV/Å (upstream: no symmetry constraint, 0.05 eV/Å); a compound whose relaxation (atoms and cell)
  does not converge gets no prediction (upstream kept unconverged structures);
- it uses the supercell matrix, primitive matrix and displacements (0.01 Å) of the DFT calculation
  (upstream: supercells at least 20 Å long, 0.015 Å), i.e. supercells of at most 180 atoms instead of
  3,822;
- C_V is taken on the 20 x 20 x 20 q-mesh of the reference (upstream: phonopy's default mesh), and the
  dynamical stability is judged as in the reference (imaginary modes below -50 K at the q-points
  commensurate with the supercell); summaries also cover the 996 compounds that are stable in DFT.

Changed on purpose (the numbers of a successful run are not affected):

- The Equilibrium benchmark displaces the atoms with a generator seeded by `seed`; upstream used an
  unseeded generator, so its results changed from run to run.
- The Phonon benchmark builds compact force constants and no eigenvectors (the heat capacity is
  identical; memory is much smaller).
- Checkpoints are JSON files with honest extensions and can resume any benchmark.
- Bug fixes: `PhononBenchmark` options raised `TypeError`; `SofteningBenchmark` could not be combined with
  other benchmarks and returned only `None` when given a model name; subsampling changed the global
  `random` state (the same subsets are still drawn); results of several models were joined on row
  position instead of the material id; importing matcalc queried the network; an Equilibrium run crashed
  at the end (`AttributeError` in the fingerprint step) as soon as one relaxation had not converged.

Kept on purpose, for comparability with published numbers (Equilibrium, Elasticity):

- Convergence of a relaxation is judged on the atomic forces only (not on the residual cell stress).
- Equilibrium atoms are displaced by a random distance of at most 0.1 Å (pymatgen's `Structure.perturb`
  with `min_distance=0`), not by exactly 0.1 Å.

## Citing

Please cite matcalc (see [`citation.cff`](citation.cff)) and the papers of the datasets and models you use;
for the softening benchmark, B. Deng et al., npj Comput. Mater. 11, 9 (2025); for the phonon benchmark,
A. Loew et al., npj Comput. Mater. (2025), doi:10.1038/s41524-025-01650-1; for the discovery benchmark,
J. Riebesell et al., Nat. Mach. Intell. 7, 836 (2025), doi:10.1038/s42256-025-01055-1, and H.-C. Wang,
S. Botti, M. A. L. Marques, npj Comput. Mater. 7, 12 (2021); for the kappa benchmark, B. Póta et al.,
arXiv:2408.00755, and A. Togo, L. Chaput, I. Tanaka, Phys. Rev. B 91, 094306 (2015).
