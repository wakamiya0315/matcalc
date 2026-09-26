# The four benchmarks

All four compare a machine-learning interatomic potential (MLIP) with DFT (PBE) reference data from the
Hugging Face dataset [`materialyze/matcalc-bench`](https://huggingface.co/datasets/materialyze/matcalc-bench).
Units follow ASE: energies in eV, forces in eV/Å, stresses in eV/Å³; moduli are reported in GPa.

Every benchmark is a sequence of stages over *all* materials of a chunk, and only two operations touch the
MLIP: `relax` and `single_point` of a simulator (`src/matcalc/simulation/`). Relaxations use the FIRE
optimizer on a Frechet cell filter, so atomic positions and the cell relax together. A relaxation counts as
converged when the largest force on any atom is at most `fmax` (the residual cell stress is not checked,
as in upstream matcalc).

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

Dataset: `alexandria-binary-pbe-phonon-2025.1.json.gz`, 1,170 binary compounds (primitive cells of 2–180
atoms).

1. **Relaxation** of atoms and cell (0.05 eV/Å, at most 5000 steps). The relaxed structure is used even if
   the relaxation did not converge; `status` then reads `ok (relaxation not converged)`.
2. **Displacements.** phonopy builds a supercell that repeats the cell `ceil(L / |a_i|)` times along each
   lattice vector a_i and displaces each symmetry-distinct atom by 0.015 Å (`symprec` = 1e-5 Å); the
   dataset needs about 35,000 displaced supercells. L = `min_supercell_length` is 15 Å (median 270 atoms,
   up to 1,950); upstream matcalc uses 20 Å (median 512, up to 3,822), which `min_supercell_length=20`
   reproduces. For compounds without imaginary modes C_V at 15 Å agrees with 20 Å (and with 25 Å) within
   0.72 J/(K·mol); compounds that are dynamically unstable with the MLIP get a C_V that depends on the
   supercell whatever its size, because phonopy leaves imaginary modes out of the thermal properties
   ([validation](validation.md)). The DFT reference (Alexandria) used supercells of at least 12 Å.
3. **Forces** on every displaced supercell (single points).
4. **Thermal properties.** Force constants → phonon frequencies on phonopy's default q-point mesh →
   C_V(T) in the harmonic approximation, on a 0–1000 K grid in 10 K steps. `CV` is C_V at 300 K, in
   J/(K·mol) per mole of primitive cells (at most 3R × atoms per cell).

Columns: `CV_DFT`, `CV_<model>`, `min_frequency_<model>` (lowest frequency on the mesh in THz; negative
values are imaginary modes, i.e. the structure is dynamically unstable with this MLIP),
`relax_steps_<model>`, `status_<model>`. Summary: MAE and STDAE of `CV`.

## Softening — `SofteningBenchmark` (`benchmarks/softening.py`)

Dataset: `wbm-high-energy-states.json.gz`, 979 WBM materials with up to 10 high-energy configurations
("frames", 2–30 atoms) and their DFT forces. Reference: B. Deng et al., npj Comput. Mater. 11, 9 (2025).

1. **Forces** on every frame at its DFT geometry (single points).
2. **Softening scale.** The slope a of the least-squares line through the origin, F_MLIP ≈ a · F_DFT, over
   all force components of a material's frames: a = Σ x·y / Σ x². A value below 1 means the MLIP's forces
   are systematically weaker than DFT's, i.e. its potential energy surface is too soft.

Columns: `softening_scale_<model>`, `status_<model>`. Summary: mean and std of `softening_scale`.
