# The four benchmarks

All four compare a machine-learning interatomic potential (MLIP) with DFT (PBE) reference data: the
Hugging Face dataset [`materialyze/matcalc-bench`](https://huggingface.co/datasets/materialyze/matcalc-bench)
(Equilibrium, Elasticity, Softening) and the packaged Phonon dataset.
Units follow ASE: energies in eV, forces in eV/Å, stresses in eV/Å³; moduli are reported in GPa.

Every benchmark is a sequence of stages over *all* materials of a chunk, and only two operations touch the
MLIP: `relax` and `single_point` of a simulator (`src/matcalc/simulation/`). Relaxations use the FIRE
optimizer on a Frechet cell filter, so atomic positions and the cell relax together. In Equilibrium and
Elasticity a relaxation counts as converged when the largest force on any atom is at most `fmax` (the
residual cell stress is not checked, as in upstream matcalc); the Phonon benchmark requires FIRE's own
criterion, every force on the atoms and on the cell below `fmax`.

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
3. **Forces** on every displaced supercell (single points).
4. **Harmonic properties.** Compact force constants → frequencies on a 20 x 20 x 20 q-point mesh → C_V at
   300 K, in J/(K·mol) per mole of primitive cells. The compound is **dynamically stable** when no
   frequency at the q-points commensurate with the supercell is below -50 K (-1.04 THz), the criterion of
   the reference.

174 compounds are dynamically unstable already in DFT (`stable_DFT` is false); phonopy leaves imaginary
modes out of C_V, so for them the value depends on details of the calculation. `summarize` reports the
errors over all compounds and over the 996 DFT-stable ones (`"CV (DFT-stable)"`), and a stability table
(`TS`/`TU`: stable/unstable in both; `FU`: stable only in DFT; `FS`: stable only with the MLIP).

Columns: `CV_DFT`, `stable_DFT`, `CV_<model>`, `stable_<model>`, `min_frequency_<model>` (lowest frequency
at the commensurate q-points, THz; negative values are imaginary modes), `relax_steps_<model>`,
`status_<model>`.

## Softening — `SofteningBenchmark` (`benchmarks/softening.py`)

Dataset: `wbm-high-energy-states.json.gz`, 979 WBM materials with up to 10 high-energy configurations
("frames", 2–30 atoms) and their DFT forces. Reference: B. Deng et al., npj Comput. Mater. 11, 9 (2025).

1. **Forces** on every frame at its DFT geometry (single points).
2. **Softening scale.** The slope a of the least-squares line through the origin, F_MLIP ≈ a · F_DFT, over
   all force components of a material's frames: a = Σ x·y / Σ x². A value below 1 means the MLIP's forces
   are systematically weaker than DFT's, i.e. its potential energy surface is too soft.

Columns: `softening_scale_<model>`, `status_<model>`. Summary: mean and std of `softening_scale`.
