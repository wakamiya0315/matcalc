# Validation

How the refactored code and the TorchSim simulator were checked against upstream matcalc, and how fast
they are. Model: MACE-MatPES-PBE-0 (float64 unless stated). Hardware: TSUBAME4, NVIDIA H100 MIG 3g.47gb
slice with 4 CPU cores (`gpu_h`); quick checks on the shared interactive node. Software: torch 2.14.0+cu126,
torch-sim-atomistic 0.6.2, mace-torch 0.3.16, ase 3.29.0, pymatgen 2026.9.23, phonopy 4.6.0. Scripts are in
[`validation/`](../validation/).

Agreement is judged material by material with these tolerances: |ΔK_vrh|, |ΔG_vrh| ≤ 1 GPa;
|ΔC_V(300 K)| ≤ 0.5 J/(K·mol); |ΔE_form| ≤ 5 meV/atom; |Δd| ≤ 0.01; |Δ softening scale| ≤ 0.01; and the same
materials without a prediction (NaN).

## 1. Refactor vs upstream (V2)

Upstream `main` and the refactored code on the same 5 random materials per benchmark (seed 42), ASE
simulator. Upstream's unseeded Equilibrium perturbation was given the same seed. "Floor" is the difference
between two runs of upstream itself (GPU summation order).

| Benchmark | Quantity | Floor (upstream vs upstream) | Refactor vs upstream |
|---|---|---|---|
| Softening | softening scale (max rel.) | 6.1e-11 | 6.4e-11 |
| Elasticity | K_vrh / G_vrh (max rel.) | 4.7e-14 / 4.8e-14 | 5.8e-14 / 3.4e-14 |
| Phonon | C_V (max abs., J/(K·mol)) | 3.9e-12 | 3.5e-12 |
| Equilibrium | E_form (eV/atom) / d (max abs.) | 2.0e-13 / 4.6e-15 | 7.2e-14 / 2.3e-15 |

(The softening floor is set by upstream's `curve_fit` tolerance; the refactor uses the closed form.)

## 2. TorchSim vs ASE for the same model (V3)

Single points on cells from the datasets, TorchSim MACE vs ASE MACE (maximum over the cells):

| Cells | n | max \|ΔE\|/atom (eV) | max \|ΔF\| (eV/Å) | max \|Δσ\| (eV/Å³) |
|---|---|---|---|---|
| Elasticity, DFT cells | 10 | 1.8e-15 | 7.3e-14 | 9.3e-16 |
| Elasticity, strained ±1 % / ±6 % | 20 | 3.6e-15 | 6.1e-14 | 2.1e-15 |
| Softening, high-energy frames | 20 | 2.8e-15 | 5.4e-14 | 3.8e-15 |
| Equilibrium, perturbed 0.1 Å | 10 | 1.4e-15 | 6.2e-14 | 2.0e-15 |
| Dense cells, 0.23–0.24 atoms/Å³ | 4 | 1.2e-14 | 1.0e-14 | 2.1e-14 |

The cell-list neighbour list gives the same numbers (≤ 8.8e-14 eV/Å).

Relaxations (FIRE + Frechet cell filter, fmax 0.05 eV/Å, 500 steps) of 30 structures, with room for only
about four structures per batch so that most of them join a batch that is already running:

| | max \|ΔE\|/atom (eV) | max \|Δ lattice\| (Å) | steps (TorchSim − ASE) | converged flag flips |
|---|---|---|---|---|
| `TorchSimSimulator` | 5.4e-12 | 2.6e-9 | 0 for all 30 | 0 |
| TorchSim's own FIRE step | 4.6e-4 | 2.0e-2 | −6 … +9 | 2 |

Three differences between TorchSim's `ase_fire` and ASE's FIRE on a `FrechetCellFilter` had to be undone to
get the first row (see `simulation/torchsim.py`): the first step of a structure that joins a running batch
(TorchSim halved its time step), the order in which FIRE updates its mixing parameter after `n_min` downhill
steps (TorchSim mixes with the already reduced value), and the convergence test (ASE tests the atomic forces
transformed by the deformation gradient plus the cell forces, and tests before the first step too).

## 3. Subsets of 100 materials (V4)

Same random subset (seed 42) with upstream `main` (ASE, `n_jobs=1`) and with `TorchSimSimulator`, one after
the other on one `gpu_h` slice. For Equilibrium the reference is the refactored ASE path (identical to
upstream, section 1), because **upstream `main` crashes at the end of the Equilibrium run as soon as one
relaxation has not converged** (`AttributeError: 'float' object has no attribute 'sites'` in the
fingerprint step).

| Benchmark | Reference time | TorchSim time | Within tolerance | Largest difference |
|---|---|---|---|---|
| Softening | 28.6 s | 18.3 s | 100 % | 2.9e-10 |
| Elasticity | 114.9 s | 42.7 s | 100 % | K 9.0e-8 GPa, G 4.3e-8 GPa |
| Equilibrium | 956 s | 342 s | 100 % | E_form 6.1e-13 eV/atom, d 2.2e-13 |

On 100 materials the TorchSim times are dominated by fixed costs (measuring the batch capacity, the first
calls); the full datasets below show the real speed-up.

## 4. Full datasets (V5)

| Benchmark | Materials | Reference (ASE) | TorchSimSimulator | Speed-up | Within tolerance | NaN pattern |
|---|---|---|---|---|---|---|
| Softening | 979 | 241 s | 34 s | 7.0× | 979/979 (max 5.7e-10) | identical |
| Elasticity | 3,953 | 5,667 s | 372 s | 15.2× | 3,953/3,953 (median 3.5e-12 GPa, max 0.15 GPa) | identical (9) |
| Equilibrium | 972 | 2,330 s | 427 s | 5.5× | 972/972 (max 4.2e-11 eV/atom) | identical (20) |
| Phonon | 1,170 | 7,070 s | *pending* | | | |

References: upstream `main` for Softening, Elasticity and Phonon; the refactored ASE path for Equilibrium
(upstream crashes, see above). TorchSim runs use float64 and 4 worker processes for phonopy and fingerprints.

Stage times of the TorchSim runs (s): Elasticity relax 149, strains 27, single points 164, fits 31;
Equilibrium elemental references 240, relax 81, fingerprints 106 (ASE path: 831, 1,030, 467).

Errors against DFT are identical for both paths: Elasticity MAE K_vrh 18.50 GPa, G_vrh 13.16 GPa
(3,944 materials); Equilibrium MAE E_form 0.0891 eV/atom, mean d 0.308 (952 materials); Softening mean
scale 0.928 (std 0.127).

## 5. Where the time goes, and options that did not pay off

Measured on one `gpu_h` slice (12 Phonon materials, 307 displaced supercells, 54–1,050 atoms, 155k atoms):

| Supercell forces | Time | max \|ΔF\| vs ASE (eV/Å) |
|---|---|---|
| ASE, float64 | 26.4 s | – |
| TorchSim, float64 | 22.7 s | 2.2e-13 |
| TorchSim, float64, cell-list neighbour list | 22.8 s | 7.0e-10 |
| TorchSim, float32 | 18.6 s | 1.2e-4 |
| TorchSim, float64, cuEquivariance kernels | 69.6 s | 7.0e-10 |

Relaxing 300 small Elasticity cells: ASE 131.9 s, TorchSim 30.2 s.

- Large supercells already keep the GPU busy one at a time, so batching them gains little; the Phonon
  speed-up comes from the batched relaxations and the parallel phonopy step. Small cells (relaxations,
  strained cells, softening frames) gain the most.
- float32 is an option (`--dtype float32`) but not the default: on the 100-material subsets it stayed within
  tolerance for Softening (max 6e-4), Elasticity (K 0.016 GPa) and Phonon (C_V 0.18 J/(K·mol)); in
  Equilibrium E_form differed by at most 0.3 meV/atom but one relaxation's convergence flag flipped.
- cuEquivariance kernels were slower on this MIG slice and float64 relaxations were not identical to the
  e3nn ones (1.6e-5 eV/atom, different step counts); the option was removed.
- TorchSim's default GPU neighbour list caps the neighbours per atom at 192 within 6 Å; dense structures in
  the full datasets exceed it, so the MACE model uses `GrowingNeighborList`, which raises the cap on demand.
