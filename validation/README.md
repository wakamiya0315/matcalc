# Validation scripts

Scripts behind [docs/validation.md](../docs/validation.md). They were run on TSUBAME4 (NVIDIA H100, MIG
3g.47gb slice) with MACE-MatPES-PBE-0 in float64 as the test model, MACE-MP-0 (`--model medium`) for the
Discovery, Kappa and Diatomics benchmarks, whose predictions Matbench Discovery publishes, and MACE-OFF23
(`--model off-medium`) for the molecular benchmarks, whose results MLIPAudit publishes. Each run selects the code under test with
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
| `mlipaudit_check.py` | sections 10–11 | Noncovalent, Conformers and Reactions runs against MLIPAudit's published results for the same model, complex by complex, molecule by molecule and reaction by reaction, and MLIPAudit's summary against `summarize` on the run. |
| `tsubame_discovery_shards.sh`, `merge_shards.py` | — | The whole Discovery benchmark as a TSUBAME job array of four shards (one `gpu_h` slice each), and the join of their tables with the metrics of the whole run. |
| `gmtkn55_check.py` | section 12 | GMTKN55Benchmark given the PBEh-3c energies that the GMTKN55 repository ships, against the repository's published PBEh-3c reaction energies and WTMAD-2. |
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

The molecular benchmarks with MACE-OFF23 (medium; MLIPAudit evaluated it in float32), compared with MLIPAudit's
result files (`https://huggingface.co/datasets/InstaDeepAI/mlipaudit-results/resolve/<revision>/MACE-OFF_ext/<benchmark>/result.json`,
`<benchmark>` = `noncovalent_interactions`, `conformer_selection` or `reactivity`):

```bash
PYTHONPATH=/path/to/main/src python run_one.py fork-torchsim noncovalent --model off-medium --dtype float32 --out noncovalent.csv
python mlipaudit_check.py noncovalent noncovalent.csv noncovalent_interactions.json
PYTHONPATH=/path/to/main/src python run_one.py fork-torchsim conformers --model off-medium --dtype float32 --out conformers.csv
python mlipaudit_check.py conformers conformers.csv conformer_selection.json
PYTHONPATH=/path/to/main/src python run_one.py fork-torchsim reactions --model off-medium --dtype float32 --out reactions.csv
python mlipaudit_check.py reactions reactions.csv reactivity.json
```

GMTKN55: the benchmark given the PBEh-3c energies that the repository ships, against its published PBEh-3c
results, then MACE-OFF23 on the reactions of its elements (and of neutral closed-shell molecules only):

```bash
PYTHONPATH=/path/to/main/src python gmtkn55_check.py
PYTHONPATH=/path/to/main/src python run_one.py fork-torchsim gmtkn55 --model off-medium --out gmtkn55.csv
PYTHONPATH=/path/to/main/src python run_one.py fork-torchsim gmtkn55 --model off-medium --neutral-closed-shell --out gmtkn55_neutral.csv
```
