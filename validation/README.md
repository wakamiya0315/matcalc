# Validation scripts

Scripts behind [docs/validation.md](../docs/validation.md). They were run on TSUBAME4 (NVIDIA H100, MIG
3g.47gb slice) with MACE-MatPES-PBE-0 in float64. Each run selects the code under test with
`PYTHONPATH=<checkout>/src`: a checkout of `main` (upstream matcalc) or of this branch.

| Script | Step | What it does |
|---|---|---|
| `v2_run_bench.py`, `v2_compare.py` | V2 | One benchmark with upstream `main` or with the refactored code (ASE, same seed), then a material-by-material comparison. Upstream's unseeded Equilibrium perturbation is given the same seed. |
| `v3_model_parity.py` | V3 | TorchSim MACE vs ASE MACE: single points on cells from the datasets (two neighbour lists) and relaxations in which structures join running batches. |
| `run_one.py` | V4, V5 | One benchmark with `upstream`, `fork-ase` or `fork-torchsim`; writes the table (CSV) and timings (JSON). |
| `compare.py` | V4, V5 | Per-material agreement of two tables against the tolerances (K, G 1 GPa; C_V 0.5 J/(K·mol); E_form 5 meV/atom; d 0.01; softening scale 0.01) and the NaN pattern. |
| `tsubame_run_v5.sh` | V5 | The `qsub` job script (one benchmark, one code path, on a `gpu_h` node). |

Example (a V4-sized subset):

```bash
PYTHONPATH=/path/to/main/src      python run_one.py upstream      elasticity --n-samples 100 --out elasticity_upstream.csv
PYTHONPATH=/path/to/torchsim/src  python run_one.py fork-torchsim elasticity --n-samples 100 --out elasticity_torchsim.csv --workers 4
python compare.py elasticity_upstream.csv elasticity_torchsim.csv
```
