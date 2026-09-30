# Validation scripts

Scripts behind [docs/validation.md](../docs/validation.md). They were run on TSUBAME4 (NVIDIA H100, MIG
3g.47gb slice) with MACE-MatPES-PBE-0 in float64 as the test model, and MACE-MP-0 (`--model medium`) for the
Discovery and Kappa benchmarks, whose predictions Matbench Discovery publishes. Each run selects the code under test with
`PYTHONPATH=<checkout>/src`: a checkout of `upstream-main` (upstream matcalc) or of `main` (this fork).

| Script | Step | What it does |
|---|---|---|
| `v2_run_bench.py`, `v2_compare.py` | V2 | One benchmark with upstream matcalc (`upstream-main`) or with the refactored code (ASE, same seed), then a material-by-material comparison. Upstream's unseeded Equilibrium perturbation is given the same seed. |
| `v3_model_parity.py` | V3 | TorchSim MACE vs ASE MACE: single points on cells from the datasets (two neighbour lists) and relaxations in which structures join running batches. |
| `run_one.py` | V4, V5 | One benchmark with `upstream`, `fork-ase` or `fork-torchsim`; writes the table (CSV) and timings (JSON). |
| `compare.py` | V4, V5 | Per-material agreement of two tables against the tolerances (K, G 1 GPa; C_V 0.5 J/(K·mol); E_form 5 meV/atom; d 0.01; softening scale 0.01) and the NaN pattern. |
| `tsubame_run_v5.sh` | V5 | The `qsub` job script (one benchmark, one code path, on a `gpu_h` node). |
| `mace_models.py` | all | The test model: MACE-MatPES-PBE-0 as an ASE calculator or a TorchSim model from the same checkpoint (not part of matcalc). |
| `phonon_dft_check.py` | V12 | The Phonon benchmark's phonopy step fed with the DFT forces of Alexandria's files: how well it gives back the DFT heat capacities and stability. |
| `v17_phonon_symmetry.py` | V17 | The Phonon benchmark on a subset in three settings (as before 2026-09-28, the default that subtracts the residual forces, `use_symmetry=False`) for one MLIP, loaded by a `module:function` given on the command line, and the comparison of the three tables. |
| `diatomics_curves.py` | section 9 | Diatomic curves of a MACE model (ASE or TorchSim) in Matbench Discovery's prediction format, compared point by point with its published curves. |
| `matbench_discovery_check.py` | sections 7–8 | Discovery and Kappa runs against Matbench Discovery's published predictions, crystal by crystal; the leaderboard's κ metrics recomputed with matcalc's functions; the packaged κ reference against the published one; MP2020 corrections recomputed with the installed pymatgen. |

Example (a V4-sized subset):

```bash
PYTHONPATH=/path/to/upstream-main/src  python run_one.py upstream      elasticity --n-samples 100 --out elasticity_upstream.csv
PYTHONPATH=/path/to/main/src           python run_one.py fork-torchsim elasticity --n-samples 100 --out elasticity_torchsim.csv --workers 4
python compare.py elasticity_upstream.csv elasticity_torchsim.csv
```

Discovery and Kappa with MACE-MP-0, compared with Matbench Discovery's files (Figshare article 22715158):

```bash
PYTHONPATH=/path/to/main/src python run_one.py fork-torchsim discovery --model medium --n-samples 10000 --workers 3 \
    --out discovery.csv --checkpoint discovery.ckpt.json.gz
python matbench_discovery_check.py discovery discovery.csv mace-mp-0-2023-12-11-discovery.csv.gz
OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 PYTHONPATH=/path/to/main/src python run_one.py fork-torchsim kappa --model medium --workers 2 --out kappa.csv
python matbench_discovery_check.py kappa kappa.csv mace-mp-0-2024-11-09-phonons-kappa-103.json.gz \
    2024-11-09-kappas-phononDB-PBE-noNAC.json.gz
```

