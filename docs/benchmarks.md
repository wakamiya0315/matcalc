# The benchmarks

All six compare a machine-learning interatomic potential (MLIP) with DFT (PBE) reference data: the
Hugging Face dataset [`materialyze/matcalc-bench`](https://huggingface.co/datasets/materialyze/matcalc-bench)
(Equilibrium, Elasticity, Softening), the packaged Phonon and Kappa datasets, and Matbench Discovery's WBM
files (Discovery).
Units follow ASE: energies in eV, forces in eV/Å, stresses in eV/Å³; moduli are reported in GPa.

Every benchmark is a sequence of stages over *all* materials of a chunk, and only two operations touch the
MLIP: `relax` and `single_point` of a simulator (`src/matcalc/simulation/`). Relaxations use the FIRE
optimizer on a Frechet cell filter, so atomic positions and the cell relax together. In Equilibrium and
Elasticity a relaxation counts as converged when the largest force on any atom is at most `fmax` (the
residual cell stress is not checked, as in upstream matcalc); the Phonon benchmark requires FIRE's own
criterion, every force on the atoms and on the cell below `fmax`. Discovery and Kappa, like Matbench
Discovery, keep the prediction of a relaxation that reaches its step limit (Discovery notes it in the
status).

`n_samples` and `seed` select a random subset (`random.Random(seed).sample`, the same subset upstream
matcalc draws with that seed).

## Equilibrium — `EquilibriumBenchmark` (`benchmarks/equilibrium.py`)

Dataset: `wbm-random-pbe52-equilibrium-2025.1.json.gz`, 972 WBM compounds (2–40 atoms).

1. **Elemental references.** For every element in the selected compounds, all candidate ground-state
   structures in `elemental_refs/MP-PBE-Element-Refs.json.gz` are relaxed (`fmax` = 0.05 eV/Å, at most 500
   steps). The lowest energy per atom of each element is its chemical potential μ_i. (With the full dataset
   this is 769 structures; carbon alone has 62.)
2. **Relaxation.** Every atom of each DFT-relaxed compound is moved in a random direction by a random
   distance of at most 0.1 Å (pymatgen `Structure.perturb`, generator seeded with `seed`), then atoms and
   cell are relaxed (0.05 eV/Å, 500 steps). A compound whose relaxation does not converge gets no prediction.
3. **Formation energy.** E_form = (E − Σ_i n_i μ_i) / N, in eV/atom.
4. **Structural distance.** `d` is the Euclidean distance between matminer `SiteStatsFingerprint`
   vectors (CrystalNN "ops" preset; mean, std, min, max over sites) of the MLIP- and DFT-relaxed structures.

Columns: `Eform_DFT`, `Eform_<model>`, `d_<model>`, `structure_DFT`, `structure_<model>`,
`relax_steps_<model>`, `status_<model>`. Summary: MAE and STDAE of `Eform`; mean and std of `d`.

## Elasticity — `ElasticityBenchmark` (`benchmarks/elasticity.py`)

Dataset: `mp-binary-pbe-elasticity-2025.1.json.gz`, 3,953 binary compounds (2–80 atoms).

1. **Relaxation** of atoms and cell (0.05 eV/Å, 500 steps). No prediction if it does not converge.
2. **Strained cells.** The relaxed cell is strained along each Voigt component: ±0.5 % and ±1 % along xx,
   yy, zz and ±3 % and ±6 % along yz, xz, xy (pymatgen `DeformedStructureSet`, 24 cells). Atoms are not
   relaxed inside the strained cells.
3. **Stresses** of the 24 strained cells and of the relaxed cell (single points).
4. **Fit.** Each elastic constant C_ij is the slope of stress component j against the Green-Lagrange strain
   component i (the relaxed cell is the zero-strain point). `K_vrh` and `G_vrh` are the Voigt-Reuss-Hill
   averages of C, in GPa. A warning is logged when the mean R² of the fits is below 0.95.

Columns: `K_vrh_DFT`, `G_vrh_DFT`, `K_vrh_<model>`, `G_vrh_<model>`, `relax_steps_<model>`,
`status_<model>`. Summary: MAE and STDAE of `K_vrh` and `G_vrh`.

## Phonon — `PhononBenchmark` (`benchmarks/phonon.py`)

Dataset: `benchmarks/data/alexandria-pbe-phonon.json.gz` (part of the package), the 1,170 binary compounds
of upstream's Phonon benchmark with the settings of their DFT phonon calculations. The DFT reference is the
Alexandria PBE phonon database (A. Loew, D. Sun, H.-C. Wang, S. Botti, M. A. L. Marques, npj Comput. Mater.
2025, doi:10.1038/s41524-025-01650-1; data at https://alexandria.icams.rub.de/data/phonon_benchmark/,
CC BY 4.0), a PBE recalculation of the MDR phonon database with phonopy. The benchmark repeats that
calculation with the MLIP, following the paper's protocol for MLIPs; `scripts/build_phonon_dataset.py`
extracts the settings from Alexandria's phonopy files.

1. **Relaxation** of the PBE unit cell of the DFT calculation (conventional cell, 2–180 atoms): FIRE on a
   Frechet cell filter with a symmetry constraint that keeps the space group (ASE's `FixSymmetry`, or
   TorchSim's), until every force on the atoms and on the cell is below 0.005 eV/Å, at most `max_steps`
   (5000) steps. A compound whose relaxation does not converge gets no prediction (`status` reads
   `relaxation not converged`).
2. **Displacements.** phonopy with the supercell matrix (diagonal, on the conventional cell; supercells of
   24–180 atoms, median 108) and the primitive matrix of the DFT calculation, and with the same displaced
   atoms and displacements (0.01 Å): 36,407 displaced supercells in total, the same cells as in DFT.
3. **Forces** on every displaced supercell and on the undisplaced one (single points).
4. **Harmonic properties.** Compact force constants from the forces of the displaced supercells minus those
   of the undisplaced one → frequencies on a 20 x 20 x 20 q-point mesh → C_V at 300 K, in J/(K·mol) per
   mole of primitive cells. The compound is **dynamically stable** when no
   frequency below -50 K (-1.04 THz) appears at the q-points where the reference looked: the points
   (n1/S1, n2/S2, n3/S3) of the (diagonal) supercell matrix S, taken, as in the reference's files, as
   reduced coordinates of the primitive reciprocal lattice.

**Symmetry.** Steps 2–4 use the crystal's symmetry, as the DFT calculation did: phonopy displaces only
symmetry-inequivalent atoms, completes the displacement–force pairs with the site symmetry of the displaced
atom and copies the force constants to equivalent atoms with the space group. Its point operations include
inversion and mirrors, so this is exact only for an MLIP whose forces transform with all of them, as DFT
forces and those of O(3)-equivariant MLIPs do. An SO(3)-equivariant or non-equivariant MLIP breaks it in two
ways ([docs/validation.md](validation.md#53-mlips-that-are-not-o3-invariant-v17)):

- The forces it leaves on the relaxed cell do not follow the cell's symmetry (the symmetry constraint of the
  relaxation symmetrizes the forces, so the relaxation stops on their symmetric part), and phonopy would
  read them as a response to the displacements. Step 4 subtracts them (`subtract_residual_forces=True`,
  the default since 2026-09-28; for an O(3)-equivariant MLIP it changes nothing, since phonopy's
  symmetrization cancels a residual that follows the symmetry). The largest of these forces is reported as
  `residual_force_<model>` (eV/Å).
- Its forces change under inversion and mirrors, so the pairs that phonopy completes with them are not
  the model's. `use_symmetry=False` avoids this: phonopy then uses the lattice translations only (they hold
  for any MLIP), every atom of the primitive cell is displaced by ±0.01 Å along the three lattice
  directions, and the force constants are solved without symmetry (six supercells per atom of the primitive
  cell, 141,300 supercells instead of 36,407). This is exact for any MLIP.

The relaxation keeps the space group in every case, as in the reference. The summary records both settings,
and a checkpoint is not resumed with other settings.

`stable_DFT` is Alexandria's stability flag: 174 compounds are unstable already in DFT. phonopy leaves
imaginary modes out of C_V, so for them the value depends on details of the calculation. The flag does not
follow from a threshold on the DFT frequencies alone (the -50 K test on them reproduces it for about 90 %
of the compounds), so `min_frequency_DFT` (the lowest DFT frequency at the same q-points) is given too.
`summarize` reports the errors over all compounds and over the 996 DFT-stable ones (`"CV (DFT-stable)"`),
and a stability table against `stable_DFT` (`TS`/`TU`: stable/unstable in both; `FU`: stable only in DFT;
`FS`: stable only with the MLIP).

Columns: `CV_DFT`, `stable_DFT`, `min_frequency_DFT`, `CV_<model>`, `stable_<model>`,
`min_frequency_<model>` (THz; negative values are imaginary modes), `residual_force_<model>` (eV/Å),
`relax_steps_<model>`, `status_<model>`.

## Softening — `SofteningBenchmark` (`benchmarks/softening.py`)

Dataset: `wbm-high-energy-states.json.gz`, 979 WBM materials with up to 10 high-energy configurations
("frames", 2–30 atoms) and their DFT forces. Reference: B. Deng et al., npj Comput. Mater. 11, 9 (2025).

1. **Forces** on every frame at its DFT geometry (single points).
2. **Softening scale.** The slope a of the least-squares line through the origin, F_MLIP ≈ a · F_DFT, over
   all force components of a material's frames: a = Σ x·y / Σ x². A value below 1 means the MLIP's forces
   are systematically weaker than DFT's, i.e. its potential energy surface is too soft.

Columns: `softening_scale_<model>`, `status_<model>`. Summary: mean and std of `softening_scale`.

## Discovery — `DiscoveryBenchmark` (`benchmarks/discovery.py`)

Dataset: Matbench Discovery's WBM files on Figshare (article 22715158, CC BY 4.0, about 150 MB, downloaded
into the cache on first use): 256,963 hypothetical crystals made by element substitution into known
prototypes (H.-C. Wang et al., npj Comput. Mater. 7, 12 (2021)) and relaxed with PBE (+U) in the Materials
Project setup; 215,488 of them have a prototype that is not in the Materials Project ("unique prototypes").
Reference: J. Riebesell et al., Nat. Mach. Intell. 7, 836 (2025).

1. **Relaxation** of every unrelaxed structure (FIRE on a Frechet cell filter, 0.05 eV/Å, at most 500
   steps). As in Matbench Discovery, a relaxation that reaches the step limit still gives a prediction.
2. **Formation energy.** E_form = (E − Σ_i n_i μ_i)/N + ΔE_MP2020, with the Materials Project elemental
   references μ_i and the MP2020 correction per atom of the relaxed structure, as in Matbench Discovery.
   The correction depends on the structure only through the oxide type (oxide, peroxide, superoxide,
   ozonide; from the O–O and O–H distances) and the sulfide type; where the relaxation changes one of them,
   the difference between the corrections of the relaxed and the DFT structure (both computed with the
   installed pymatgen) is added to the published correction of the WBM DFT entry. Recomputing the whole
   correction instead would change it for 1.4 % of the entries relative to the published DFT values with
   pymatgen 2026.9, which classifies some anions (Te, Se, Si, H, Cl) differently (matbench-discovery issue
   #358). The model's energies must be on the scale of Materials Project PBE/PBE+U calculations (as those
   of MPtrj-trained models are).
3. **Distance to the convex hull.** E_hull = E_hull(DFT) + E_form − E_form(DFT); the hull of the Materials
   Project phases stays fixed. A crystal counts as stable at E_hull ≤ 0.
4. **Geometry.** RMSD between the relaxed and the DFT-relaxed structure (pymatgen `StructureMatcher` with
   stol = 1, normalized by (volume per atom)^(1/3); 1 when the structures do not match) and the space group
   and number of symmetry operations (moyopy) at 1e-5 and 1e-2 Å, compared with those of the DFT structures
   published with the data.

Columns: `e_form_per_atom_DFT`, `e_above_hull_DFT`, `spg_num_1e-5_DFT` etc., `unique_prototype`, and per
model `e_form_per_atom`, `e_above_hull`, `rmsd`, `spg_num_1e-5`, `n_sym_ops_1e-5`, `spg_num_1e-2`,
`n_sym_ops_1e-2`, `relax_steps`, `status`. Summary (`summarize`): for all crystals and for the unique
prototypes, F1, DAF (precision over the fraction of stable crystals), precision, recall, accuracy, the
confusion counts, and MAE, RMSE and R² of E_hull, with Matbench Discovery's conventions (predictions more
than 5 eV/atom off count as missing, missing ones as unstable, energies rounded to 1 meV/atom, the DAF of
the unique prototypes divided by their unrounded fraction of stable crystals); and per tolerance, the mean
RMSD, the MAE of the number of symmetry operations and the fractions of structures whose symmetry
decreased, matched or increased.

## Kappa — `KappaBenchmark` (`benchmarks/kappa.py`)

Dataset: `data/phonondb-pbe-kappa.json.gz` (packaged; built by `scripts/build_kappa_dataset.py`, CC BY 4.0):
the 103 rock-salt, zinc-blende and wurtzite crystals of Matbench Discovery's κ_SRME task (B. Póta et al.,
arXiv:2408.00755) with their PBE unit cells, the FC2 and FC3 supercells and q-point meshes of PhononDB, and
their PBE conductivity at 300 K. The reference is recomputed from PhononDB's PBE force sets (A. Togo,
L. Chaput, I. Tanaka, Phys. Rev. B 91, 094306 (2015); MDR at NIMS) with the functions the benchmark uses
for the MLIP, so reference and prediction share the phono3py solver (see below).

1. **Relaxation** keeping the space group (FIRE on a Frechet cell filter with the symmetry constraint at
   0.01 Å, 1e-4 eV/Å, at most 300 steps). Matbench Discovery forbids cell tilts; for these cubic and
   hexagonal cells the symmetrized cell step has none anyway.
2. **Harmonic force constants** from the displaced FC2 supercells (0.01 Å, phono3py's displacements) and the
   frequencies on the q-point mesh. A crystal with imaginary modes (below 0, or below −0.01 THz for the
   acoustic modes at Γ), or whose space group (1e-5 Å) changed in the relaxation, gets no conductivity.
3. **Third-order force constants** from the displaced FC3 supercells and the lattice thermal conductivity:
   Wigner transport equation (Simoncelli, Marzari, Mauri 2019) in the relaxation-time approximation with
   isotope scattering, particle-like plus coherence conductivity, symmetrized force constants.
4. **Errors.** SRD = 2 (κ − κ_DFT)/(κ + κ_DFT) and SRE = |SRD| for κ = the mean of the diagonal; the
   mode-resolved SRME = 2 Σ_modes |κ_mode − κ_DFT,mode| / Σ_q w_q / (κ + κ_DFT), where the coherence between
   two bands is shared between them in proportion to their heat capacities. Crystals without a conductivity
   count as SRME = SRE = 2 and SRD = −2.

Matbench Discovery pins phono3py 3.30, whose Wigner solver ("MS-SMM19") phono3py 4 replaced ("SMM19") and
which needs phonopy 3.5, while the rest of this package needs phonopy 4. On PhononDB's PBE force sets the
two solvers give conductivities that differ by 0.3 % on average, and by up to 3.4 % for halides whose
coherence conductivity is large (their particle-like parts agree within 0.3 %; see `docs/validation.md`),
which is why the reference is recomputed with the solver that the benchmark uses.

The conductivities, on the CPU, take almost all of the time (about 99 % of a crystal's work) and scale with
the cores: phono3py ≥ 4.7 computes them with its Rust backend, whose threads are set by `RAYON_NUM_THREADS`
(`OMP_NUM_THREADS` only sets those of its C code); set both so that `workers` × threads fits the cores.
Since the MLIP's part (relaxations and single points) is short, the conductivities can run on another node
(for MACE-MP-0 on TSUBAME4: 19 min on an H100 MIG slice, then 44 min on 16 cores, instead of 2 h 45 min on
the slice with its 4 cores):

```python
KappaBenchmark().save_forces(model, "kappa-forces.pkl")  # GPU node: steps 1-3, saves the phono3py inputs
table = KappaBenchmark(workers=8).run_saved_forces("kappa-forces.pkl", "my-mlip")  # CPU node: step 4
```

`run_saved_forces` returns the same table as `run` (the tests check it) and refuses a file saved for another
draw or other settings.

Columns: `kappa_DFT`, and per model `kappa`, `srd`, `sre`, `srme`, `relax_steps`, `status` (`ok` or
`censored: <reason>`). Summary: κ_SRME, κ_SRE, κ_SRD (means over the crystals), the failure rate and the rate
of imaginary modes.
