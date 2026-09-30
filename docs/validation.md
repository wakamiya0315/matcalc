# Validation

How the refactored code and the TorchSim simulator were checked, and how fast they are. The MLIP of the
runs is MACE-MatPES-PBE-0 in float64, used only as a test model (it is not part of matcalc; see
`validation/mace_models.py`); sections 7 and 8 use MACE-MP-0 instead, whose predictions Matbench Discovery
publishes material by material. Hardware: TSUBAME4, NVIDIA H100 MIG 3g.47gb slice with 4 CPU cores
(`gpu_h`); quick checks on the shared interactive node. Software: torch 2.14.0+cu126, torch-sim-atomistic
0.6.2, moyopy 0.20.0, mace-torch 0.3.16, ase 3.29.0, pymatgen 2026.9.23, phonopy 4.6.0. Scripts are in
[`validation/`](../validation/).

"Upstream `main`" below is materialyzeai/matcalc at `b04715d` (2026-09-09), kept in this fork as the branch
`upstream-main`.

Agreement is judged material by material with these tolerances: |ΔK_vrh|, |ΔG_vrh| ≤ 1 GPa;
|ΔC_V(300 K)| ≤ 0.5 J/(K·mol); |ΔE_form| ≤ 5 meV/atom; |Δd| ≤ 0.01; |Δ softening scale| ≤ 0.01; and the same
materials without a prediction (NaN).

Sections 1–4 cover Equilibrium, Elasticity and Softening, which reproduce upstream, and the upstream form
of the Phonon benchmark (20 Å supercells) that was used to validate the simulators. Section 5 covers the
Phonon benchmark as it is now, on the protocol of its DFT reference.

## 1. Refactor vs upstream (V2)

Upstream `main` and the refactored code on the same 5 random materials per benchmark (seed 42), ASE
simulator. Upstream's unseeded Equilibrium perturbation was given the same seed. "Floor" is the difference
between two runs of upstream itself (GPU summation order).

| Benchmark | Quantity | Floor (upstream vs upstream) | Refactor vs upstream |
|---|---|---|---|
| Softening | softening scale (max rel.) | 6.1e-11 | 6.4e-11 |
| Elasticity | K_vrh / G_vrh (max rel.) | 4.7e-14 / 4.8e-14 | 5.8e-14 / 3.4e-14 |
| Phonon (upstream protocol) | C_V (max abs., J/(K·mol)) | 3.9e-12 | 3.5e-12 |
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

With the symmetry constraint (Phonon benchmark), TorchSim's `FixSymmetry` takes the place of ASE's; ASE's
constraint first snaps the structure onto its symmetry, and `TorchSimSimulator` does the same with ASE's
function. `BatchedFixSymmetry` symmetrizes all structures of a batch at once and agrees with TorchSim's
`FixSymmetry` to 1e-12. Section 5 compares the two paths on the full Phonon benchmark.

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

## 4. Full datasets (V5)

| Benchmark | Materials | Reference (ASE) | TorchSimSimulator | Speed-up | Within tolerance | NaN pattern |
|---|---|---|---|---|---|---|
| Softening | 979 | 241 s | 34 s | 7.0× | 979/979 (max 5.7e-10) | identical |
| Elasticity | 3,953 | 5,667 s | 372 s | 15.2× | 3,953/3,953 (median 3.5e-12 GPa, max 0.15 GPa) | identical (9) |
| Equilibrium | 972 | 2,330 s | 427 s | 5.5× | 972/972 (max 4.2e-11 eV/atom) | identical (20) |
| Phonon, upstream protocol | 1,170 | 7,070 s | 5,404 s | 1.3× | 1,169/1,169 (max 4.7e-5 J/(K·mol)) | identical (1) |

References: upstream `main` for Softening, Elasticity and Phonon; the refactored ASE path for Equilibrium
(upstream crashes, see above). TorchSim runs use 4 worker processes for phonopy and fingerprints. Stage
times of the TorchSim runs (s): Elasticity relax 149, strains 27, single points 164, fits 31; Equilibrium
elemental references 240, relax 81, fingerprints 106 (ASE path: 831, 1,030, 467). Section 6 gives the times
of the final code.

The upstream Phonon protocol needs the forces on 35,000 displaced supercells of up to 3,822 atoms (27
million atoms); a supercell of several hundred atoms already keeps the GPU busy on its own, so batching gains
little there. Validating it exposed a batching bug: the batch capacity was raised to the largest structure
of a call, so in the chunk with the 3,822-atom SiC supercells (which do not fit on the 47 GB GPU at all)
every other batch ran out of memory and was computed twice. Single points now keep the measured capacity,
evaluate a larger structure on its own, and lower the capacity after running out of memory.

Errors against DFT are identical for both paths: Elasticity MAE K_vrh 18.50 GPa, G_vrh 13.16 GPa
(3,944 materials); Equilibrium MAE E_form 0.0891 eV/atom, mean d 0.308 (952 materials); Softening mean
scale 0.928 (std 0.127); Phonon (upstream protocol) MAE C_V 13.13 J/(K·mol).

## 5. Phonon on the protocol of the DFT reference (V10–V16)

### 5.1 The reference

The DFT reference of the Phonon benchmark is the Alexandria PBE phonon database (A. Loew et al., npj
Comput. Mater. 2025, doi:10.1038/s41524-025-01650-1): a PBE recalculation of the MDR phonon database with
phonopy. Its per-compound phonopy files (https://alexandria.icams.rub.de/data/phonon_benchmark/pbe/, CC BY
4.0) give, for the 1,170 compounds of the benchmark:

- the PBE unit cell (conventional; 1–4 times the primitive cell), with the same C_V(300 K) as upstream's
  dataset (to 5e-7 J/(K·mol));
- a diagonal supercell matrix on that cell (supercells of 24–180 atoms, median 108; no fixed length:
  the shortest supercell vector ranges from 3.1 to 19.8 Å), and phonopy's primitive matrix;
- the displacements: 0.01 Å, 36,407 in total; phonopy's symmetry tolerance 1e-5 Å (phonopy 2.29/2.32);
- the frequencies at the q-points of the stability test, and (in the summary table) a stability flag.

The paper relaxes the MLIPs from the PBE geometry with ASE's FIRE on a Frechet cell filter keeping the
space group, fmax 0.005 eV/Å, and takes the thermal properties on a 20 x 20 x 20 q-mesh at 300 K. The
benchmark follows this and reuses the DFT supercells and displacements (`scripts/build_phonon_dataset.py`
extracts them). phonopy 4.6 would not generate the same displacements from the same cells (15 of the first
40 compounds), which is another reason to take them from the files.

**The benchmark's phonopy step with the DFT forces** (`validation/phonon_dft_check.py`): for all 1,170
compounds, C_V(300 K) agrees with the DFT value to a median of 0.0076 J/(K·mol) (99 % within 0.43, largest
1.84, 9 compounds above 0.5, 6 of them DFT-unstable). phonopy's own mesh sum with the stored DFT force
constants leaves residuals of the same size on the 48 compounds tried, whatever the mesh variant, so the
reference integrated slightly differently. The frequencies stored with the reference are reproduced at the points (n/S) of the supercell
matrix taken as reduced coordinates of the primitive reciprocal lattice (96 % of the compounds with a
non-primitive unit cell within 0.05 THz; the transformed points give 31 %), so the stability test uses
these points. Alexandria's stability flag, however, does not follow from any threshold on these
frequencies: the -50 K test on the DFT frequencies reproduces it for about 90 % of the compounds. The
benchmark keeps the flag as `stable_DFT` for the DFT-stable subset and also reports `min_frequency_DFT`.

**Symmetry tolerance.** With ASE's default tolerance of 0.01 Å, 29 of the 1,170 DFT structures are found
in a higher space group than DFT's (for example P2_1/m → Pnma), and ASE's constraint snaps them onto it;
their DFT displacements then no longer fit and phonopy cannot build the force constants. At the DFT's
tolerance, 1e-5 Å, spglib and moyopy both find the DFT space group for every structure, so the benchmark
uses 1e-5 Å.

### 5.2 MACE-MatPES-PBE-0 on the full benchmark

Full benchmark, same protocol, TorchSim (final code, 3 worker processes) and ASE (one structure at a time,
chunks of 20 compounds, 4 workers), one `gpu_h` slice each:

| | Wall time | Relax | Displacements | Single points | phonopy (waiting) |
|---|---|---|---|---|---|
| ASE | 4,349 s | 2,951 s | 203 s | 1,065 s | 88 s |
| `TorchSimSimulator` | **1,205 s** | 651 s | 21 s | 508 s | 24 s |

TorchSim is 3.6× faster than ASE on this protocol, and 5.9× faster than upstream's own Phonon benchmark
with ASE (7,070 s, section 4), which needed 8 times as many supercell atoms.

**TorchSim vs ASE, compound by compound** (same code, runs V12): C_V within tolerance for 1,170/1,170
(median difference 1e-11 J/(K·mol), largest 1.4e-3), the same status and the same stability for every
compound, and the same number of FIRE steps for 1,168; the other two relaxations are among the longest
(1,081 vs 1,099 and 2,340 vs 2,339 steps). The final code passes the unit cell to phonopy as the primitive
cell when the DFT calculation did (phonopy 4 otherwise searches for a primitive cell itself); this changed
C_V of one compound (P2Se5) by 0.08 J/(K·mol) and no other by more than 0.001. For three compounds phonopy
finds a higher space group in the relaxed structure than in DFT and generates the displacements anew
(same amplitude); their C_V is the same either way.

**Results for MACE-MatPES-PBE-0** (final run):

| | |
|---|---|
| Relaxations converged within 5,000 steps (atoms and cell below 0.005 eV/Å) | 1,170/1,170 |
| MAE C_V vs DFT, all compounds | 11.71 J/(K·mol) (STDAE 20.01) |
| MAE C_V vs DFT, DFT-stable compounds (996) | 10.85 J/(K·mol) (STDAE 18.04) |
| Stability vs `stable_DFT`: TS / TU / FU / FS | 984 / 38 / 12 / 136 |

(The -50 K test on the DFT frequencies themselves agrees with `stable_DFT` for 1,057 compounds, so part
of the disagreement is in the flag.)

**Maximum number of FIRE steps.** The relaxations take a median of 40 steps (90 % within 145, 99 % within
656, the longest 2,349). With `max_steps` = 500, 20 compounds (1.7 %) would get no prediction, with 1,000
still 4; the paper reports 0.1–0.9 % failed relaxations for its models. With in-flight batching the slow
relaxations run next to the others: the last 60 compounds cost about 160 s of the relaxation stage, so a
smaller limit would save little time. The default stays at 5,000.

### 5.3 MLIPs that are not O(3)-invariant (V17)

**What was seen.** Twelve MLIPs of about a million parameters trained on MatPES (PBE and r2SCAN; six
architectures, with the UnifyMLIP library) ran the full benchmark with the protocol above (float64). The
two SO(3)-equivariant architectures, SevenNet without parity and eSEN, called 267–287 of the 996
DFT-stable compounds unstable (FU), the four O(3)-equivariant ones (MACE, NequIP, Nequix, Allegro)
13–37. The imaginary frequencies of the SO(3) models were large: a median of −8 to −15 THz over those
compounds, against −1.4 to −1.7 THz, close to the −1.04 THz threshold, for the O(3) models. Their C_V
errors stayed within or somewhat above the range of the O(3) models (PBE 7.2 and 8.2 J/(K·mol) against
7.1–10.1; r2SCAN 8.2 and 6.9 against 4.9–6.1). For comparison, Loew et al. (Table 3) report 5–23 %
false-unstable compounds for their O(3) models (MACE-MP-0, MatterSim-v1, M3GNet, CHGNet), 19 % for the SO(3)
SevenNet-0 and 85–93 % for ORB and eqV2-M, which they attribute to forces that are not energy gradients;
the paper does not discuss symmetry.

**Why** (phonopy 4.7, the traditional force-constant solver). phonopy fills the displacement–force pairs of
a displaced atom with its site symmetry, solves Φ = −F/u by pseudo-inversion and copies Φ to the
equivalent atoms with the space group. Both steps use inversion and mirrors, and both take the force on a
displaced supercell for the response to the displacement. An SO(3)-equivariant model breaks this twice:

1. Its forces on the relaxed cell do not vanish. The symmetry constraint of the relaxation symmetrizes
   the forces, so FIRE stops when their symmetric part is below `fmax`, while the model's own forces keep a
   part that does not follow the space group (median 0.006–0.013 eV/Å on the compounds below, up to
   0.12 eV/Å; 0.002–0.005 eV/Å, following the symmetry, for the O(3) models). With 0.01 Å displacements a
   residual force F₀ becomes a force-constant error of order F₀/u, 0.1–10 eV/Å², which is enough for
   imaginary modes of several THz. A residual that follows the symmetry cancels in phonopy's
   symmetrization, which is why O(3) models (and DFT) do not see it.
2. Its forces change under inversion and mirrors (by 0.003–0.09 eV/Å on displaced supercells), so the
   pairs that phonopy completes with these operations are not the model's.

The first effect is removed by subtracting the forces on the undisplaced supercell from those of every
displaced supercell (`subtract_residual_forces`, now the default: one more supercell per compound). Both
are removed by building the force constants without symmetry (`use_symmetry=False`): phonopy still uses the
lattice translations, which hold for any MLIP, and displaces every atom of the primitive cell by ±0.01 Å
along the three lattice directions (141,300 supercells for the whole dataset instead of 36,407, 3.9 times
as many; the ± pairs cancel the residual forces too). The relaxation keeps the space group in every case.

**Check** (`validation/v17_phonon_symmetry.py`, one `gpu_h` slice, float64). Four PBE models of the set
above, on 50 compounds: 30 drawn at random (seed 42) and 20 of the 222 compounds that both SO(3) models
called unstable while all four O(3) models and DFT called them stable. Three settings: the protocol before
this change (`subtract_residual_forces=False`), the new default, and `use_symmetry=False`; the differences
are against `use_symmetry=False`, which is exact for any model.

| Model | Setting | FU (of 46 DFT-stable) | MAE C_V (J/(K·mol)) | \|ΔC_V\| median / max | \|Δ lowest frequency\| max (THz) |
|---|---|---:|---:|---|---:|
| eSEN (SO(3)) | before | 25 | 7.85 | 3.16 / 39.7 | 52.4 |
| | default | 0 | 5.85 | 0.016 / 2.18 | 0.29 |
| | `use_symmetry=False` | 0 | 5.77 | – | – |
| SevenNet (SO(3)) | before | 27 | 7.46 | 0.44 / 30.3 | 35.2 |
| | default | 2 | 7.52 | 0.0014 / 0.49 | 0.13 |
| | `use_symmetry=False` | 2 | 7.54 | – | – |
| MACE (O(3)) | before | 1 | 7.22 | 1e-12 / 0.0011 | 0.0009 |
| | default | 1 | 7.22 | 6e-13 / 0.0011 | 0.0009 |
| | `use_symmetry=False` | 1 | 7.22 | – | – |
| NequIP (O(3)) | before | 2 | 8.39 | 1e-12 / 0.0015 | 0.0005 |
| | default | 2 | 8.39 | 8e-13 / 0.0015 | 0.0005 |
| | `use_symmetry=False` | 2 | 8.39 | – | – |

- For the O(3) models the three settings agree to 0.002 J/(K·mol) and 0.001 THz: subtracting the residual
  forces changes nothing, and phonopy's symmetry is exact, as expected. The numbers of section 5.2
  (MACE-MatPES-PBE-0, O(3)) were computed before the change; they are not expected to move.
- For the SO(3) models the subtraction removes the spurious instabilities (FU 25 → 0 and 27 → 2; the two
  left are unstable without symmetry too) and brings C_V within 0.5 J/(K·mol) of the exact value for 47 of
  the 50 compounds with eSEN (the other three up to 2.2 J/(K·mol), ClF) and for all 50 with SevenNet: what
  remains is the second effect. `use_symmetry=False` is the setting for such models when the numbers should
  be exact; it needs about four times as many single points.

## 6. Where the time goes

Final code, one `gpu_h` slice per benchmark, 3 worker processes (runs V16; wall time of the run without
loading the dataset):

| Benchmark | Wall time | Stages (s) | V5 code (section 4) |
|---|---|---|---|
| Softening | 27 s | single points 27 | 34 s |
| Elasticity | 255 s | relax 119, strains 3, single points 131, fits 2 | 372 s |
| Equilibrium | 253 s | relax 83, elemental references 169, fingerprints 0 | 427 s |
| Phonon | 1,205 s | relax 651, displacements 21, single points 508, phonopy 24 | — |

- **The GPU stages** (relaxations, single points) run at the throughput of the model: doubling the batch
  capacity speeds single points up by only 3–9 %, and four times the capacity runs out of memory.
- **Batch capacity.** TorchSim measures how many copies of the smallest and of the largest structure of a
  call fit on the GPU and sizes every batch for the worse of the two. For the smallest structure that is
  mostly a fixed cost per structure and per atom, which TorchSim's memory metric (atoms x density) leaves
  out: one of the Equilibrium benchmark's elemental references, a single atom in 600 Å^3, held the batches
  of all 769 references to 8,857 metric units, a tenth of what fits.
  `TorchSimSimulator` turns the same two probes into an upper bound on the memory of each structure
  (memory taken as linear in structures, atoms and metric) and uses the largest capacity with which no
  batch of the call's structures can exceed the probed memory (`memory_shares` and `batch_capacity` in
  `simulation/torchsim.py`). The references now relax with a capacity of 87,700 (219 → 170 s) and
  Elasticity's single points take 131 s instead of 144 s; Softening and Phonon change by less than 10 s.
  No batch ran out of memory. The results agree with the previous runs:
  Softening to 2e-14, Equilibrium E_form to 4e-12 eV/atom and d to 6e-13, Phonon C_V to 9e-4 J/(K·mol)
  with the same stability for every compound, Elasticity K and G to 0.17 GPa (median 1e-12). One
  Elasticity relaxation (mp-27954) is still moving when it reaches the 500-step limit; whether its largest
  force is below fmax at that step differs from run to run, also between two runs of the same code.
- **CPU work next to the GPU.** Elasticity builds the strained cells directly as ASE structures from one
  set of 24 deformation gradients (2 s for 3,953 compounds instead of 27 s through pymatgen objects) and
  runs the stress-strain fits in the worker processes while the GPU computes the next part of the single
  points (31 → 2 s of waiting). Equilibrium computes the fingerprints of the DFT structures in the workers
  while the GPU relaxes the compounds, and those of the relaxed compounds while it relaxes the elemental
  references, which therefore come after the first compounds: the fingerprint stage went from 106 s to
  nothing (E_form and d unchanged to 3e-12). The relaxation stage of the compounds includes TorchSim's
  memory probes, now made separately for the compounds and for the references.

Phonon (TorchSim, 1,170 compounds, one `gpu_h` slice):

- **Relaxation** to 0.005 eV/Å with the symmetry constraint. A FIRE step for a batch of 90 unit cells
  (2,187 atoms) takes 0.41 s in the MACE forward pass (energy, forces, stress); TorchSim's `FixSymmetry`
  added 0.14 s per step because it symmetrizes structure by structure in Python. `BatchedFixSymmetry` does
  it for the whole batch at once: 0.53 → 0.32 s per step.
- **Single points** on the 36,407 displaced supercells (3.4 million atoms): about 7,000 atoms/s, the
  throughput of the model on this GPU.
- **phonopy** (force constants, 20 x 20 x 20 mesh) runs in the worker processes while the GPU evaluates the
  next part of the supercells, so it no longer adds to the wall time. The mesh frequencies are computed a
  few hundred q-points at a time: phonopy's own mesh calculation holds the dynamical matrices of all
  q-points at once, up to 5 GB for one large low-symmetry compound, which let a 9 GB CPU job run out of
  memory with four workers. With four CPU cores, three workers leave one core to the process that drives
  the GPU.

Model-specific speed-ups (lower precision, custom kernels) are left to whoever builds the model; the
benchmark does not choose them.

## 7. Discovery with MACE-MP-0

The Discovery benchmark is compared crystal by crystal with the MACE-MP-0 predictions that Matbench
Discovery publishes (`mace-mp-0-2023-12-11-discovery.csv.gz` on Figshare, formation energies rounded to
1e-4 eV/atom), made with the same model (mace-torch's "medium" checkpoint, float64) and the same
relaxation (FIRE on a Frechet cell filter, 0.05 eV/Å, at most 500 steps). TorchSim, 3 worker processes,
one `gpu_h` slice.

**TorchSim vs ASE** (300 random crystals, seed 42): the same number of FIRE steps for every crystal,
E_form within 5.5e-6 eV/atom (median 1e-15), RMSD within 6e-11, and the same space groups except for one
crystal at 1e-5 Å (Na2Ni2O8, whose relaxation also moves by 5e-6 eV/atom between two TorchSim runs; the
same space group at 1e-2 Å). Wall time without loading the data: 225 s with ASE, 44 s with TorchSim.

**Against the published predictions.** On 10,000 random crystals, E_form agrees within 1.1e-4 eV/atom
(the rounding of the published values) for all 10,000. Before the benchmark corrected the relaxed
structure's oxide type, 14 of them differed by 0.07–0.42 eV/atom: oxides whose relaxation moves O atoms
across a threshold of the O–O distance that pymatgen uses to tell oxides, peroxides and superoxides apart
(for example BaYbO4, a peroxide in DFT and a superoxide after the MACE relaxation), so that the MP2020
correction changes. Recomputing the whole correction with the installed pymatgen instead changes it for 56
of 4,000 random DFT entries (1.4 %: anion corrections of Te, Se, Si, H or Cl; matbench-discovery issue #358).
The correction adds 4 s to the 43 s of the geometry stage.

On all 256,963 crystals (a run of the whole set, whose 37,580 crystals with O or S were run again after the
correction was added), E_form agrees within 1.5e-4 eV/atom for all but 46 of the 256,925 published
predictions, within 0.01 eV/atom for all but 2. The 46 are long relaxations: 5 reach the 500-step limit and
most others take 150–350 steps on flat energy surfaces, where the last digits of the forces decide the path
(37 of the rerun crystals also took a different number of steps than in the first run of the same code).
The largest difference, SeOCl2 (wbm-1-43995, 0.42 eV/atom), is such a case: the TorchSim and ASE paths
agree within 1e-9 Å for 45 steps, then part; at step 57 TorchSim's structure meets fmax in a flat region of
the energy surface, while ASE's, 3e-3 Å away, does not and continues to a minimum 0.42 eV/atom lower (the
published value; ASE takes 135 steps to it on a CPU, 125 on the GPU).

| | This benchmark | Leaderboard |
|---|---|---|
| All crystals: F1 / DAF / precision / recall | 0.6678 / 3.400 / 0.5835 / 0.7805 | 0.668 / 3.4 / 0.583 / 0.781 |
| All crystals: MAE / RMSE / R² of E_hull (eV/atom) | 0.0547 / 0.0988 / 0.6984 | 0.055 / 0.099 / 0.698 |
| All crystals: TP / FP / TN / FN, missing | 34,416 / 24,569 / 188,302 / 9,676, 38 | 34,420 / 24,576 / 188,295 / 9,672, 38 |
| Unique prototypes: F1 / DAF / precision / recall | 0.6694 / 3.777 / 0.5774 / 0.7964 | 0.669 / 3.777 / 0.577 / 0.796 |
| Unique prototypes: MAE / RMSE / R² | 0.0569 / 0.1013 / 0.6972 | 0.057 / 0.101 / 0.697 |
| Mean RMSD; symmetry match at 1e-5 / 1e-2 Å (249,254 crystals) | 0.0915; 0.7385 / 0.8112 | 0.0915; 0.7385 / 0.8112 |
| Symmetry decrease / increase at 1e-5 Å (249,254 crystals) | 0.0335 / 0.2239 | 0.0335 / 0.2239 |

The leaderboard's geometry metrics cover the 249,254 crystals of Matbench Discovery's file of relaxed
MACE-MP-0 structures (7,709 crystals, in contiguous ranges of ids, are missing from it); on those crystals
the RMSD agrees within 1e-4 for 99.98 % of the 231,562 that both match to their DFT structure, the same
17,688 are unmatched in both (4 in only one), and the space groups agree for 99.98 % at 1e-5 Å and 99.99 %
at 1e-2 Å (the published analysis used moyopy 0.4.2). Over all 256,963 crystals the benchmark gives a mean
RMSD of 0.0924 and symmetry matches of 0.7371 and 0.8110.

**Time.** The whole set on one `gpu_h` slice (TorchSim, 3 workers): 3.4 h, of which 11,265 s relaxation
(23 crystals per second), 975 s geometry in the worker processes, and 56 s to load the model and the data
(only the drawn crystals are parsed: 24 s for a draw of 10,000). At ASE's rate on the 300-crystal subset
(1.3 crystals per second), the whole set would take about 54 h.

## 8. Kappa with MACE-MP-0

**The metric.** With Matbench Discovery's published MACE-MP-0 mode conductivities
(`mace-mp-0-2024-11-09-phonons-kappa-103.json.gz`) and its published DFT reference, the benchmark's
`symmetric_relative_mean_error` and `symmetric_relative_difference` give κ_SRME 0.6823, κ_SRE 0.4710 and
κ_SRD −0.2845 with 3 failures: the leaderboard's values.

**The reference.** Matbench Discovery computed its DFT reference with phono3py 3.30 from PhononDB's PBE
force sets. With phono3py 3.30 (and phonopy 3.5) the benchmark's steps on those force sets reproduce it:
wurtzite AgI +0.003 % (mode-resolved SRME against the published reference 5.6e-4), BeTe +0.02 %.
phono3py 4.7, whose Wigner solver ("SMM19") the package uses, gives +0.43 % and −0.35 % (SRME 5.7e-3 and
4.2e-3) and is ten times faster (52 s instead of 548 s for AgI on the same laptop).
On all 103 compounds, the recomputed reference (phono3py 4.7, the packaged dataset; 46 min on a 10-core
laptop) differs from the published one by a median of −0.13 % (mean |Δ| 0.31 %, 99 compounds within 1 %),
and the mode-resolved SRME between the two is 0.0046 on average (at most 0.034): the floor that the solver
change puts under κ_SRME. The four compounds beyond 1 % are low-conductivity halides whose coherence
(wave-like) conductivity is a large part of the total: rock-salt AgCl +3.4 %, wurtzite and zinc-blende CuCl
+2.7 %, rock-salt AgBr +2.3 %. The difference is in the coherence term: for AgCl the particle-like
conductivity is 0.2905 W/(m K) with phono3py 4.7 and 0.2913 with 3.30, the coherence conductivity 0.2548 and
0.2360; for zinc-blende CuCl 1.3735 and 1.3773, and 0.1802 and 0.1338.

**TorchSim vs ASE** (5 random crystals, seed 42, 2 workers with 2 OpenMP threads each): the same status and
the same number of FIRE steps for every crystal, κ within 0.03 % for LiCl, LiF and CaS and 0.12 % for
ZnTe. BeTe differs by 3 % (120.4 vs 124.1 W/(m K)), but so do two runs of the same code: ASE gave 120.3
and 124.3, TorchSim 120.8 and 122.1. In MACE-MP-0's BeTe, the highest optical mode at the q-point next to Γ
(10.70 THz) has a three-phonon linewidth of 1.3e-7 THz, so its lifetime comes from isotope scattering
(2.0e-7 THz), which the tetrahedron integration over this flat band changes to 4.2e-7 THz when the forces
change by 1e-12 eV/Å; that one mode moves κ by 3 W/(m K). With the DFT forces, the same noise changes
BeTe's κ by less than 2e-4 (relative). Wall time for the 5 crystals: 365 s with ASE, 292 s with TorchSim,
of which 276 and 213 s are spent waiting for the conductivities (CPU).

**MACE-MP-0 on the 103 crystals**, in two stages (`save_forces` on one `gpu_h` slice, 19 min;
`run_saved_forces` with 8 workers of 2 threads on `cpu_16`, 44 min and 40,300 CPU-seconds; `run` on the
slice alone, with its 4 cores, takes 2 h 45 min, 2 h 25 min of it waiting for the conductivities): every
crystal relaxes (at most 81 FIRE steps)
and gets a conductivity, without imaginary modes. κ_SRME 0.661, κ_SRE 0.452, κ_SRD −0.181; on the 100
crystals that the published run completed (its other 3 failed), κ_SRME 0.663 against the published 0.643.
Crystal by crystal, κ is a median 0.5 % above the published value, within 5 % for 79 of the 100 crystals.

The largest differences are of two kinds. Some crystals are ill-conditioned, like BeTe above: the same
saved force sets give BeSe a κ of 35.6 W/(m K) on a laptop (Apple M4) and 4.4 × 10⁶ W/(m K) on the
TSUBAME node (published: 40.2), one mode being nearly undamped; the run on the slice alone, whose forces
differ from the other run's by GPU rounding, gives 35.5. Otherwise the two TSUBAME runs agree within 0.1 %
for 89 crystals and within 1 % for all (κ_SRME 0.651 and 0.661, the difference being BeSe's). Other crystals
differ from the published run in their force sets: for NaF
(74.6 against the published 50.3 W/(m K)) and BeO (500 and 622 against 313 and 390), the same MACE forces
give the same κ with phono3py 3.30 as with 4.7 (NaF: 74.72 and 74.58), and the harmonic frequencies agree
with the published ones (medians within 0.1 %), so the difference lies in the anharmonic force constants
of Matbench Discovery's 2024 run, whose forces are not published.

