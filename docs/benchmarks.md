# The benchmarks

The first seven compare a machine-learning interatomic potential (MLIP) with DFT (PBE) reference data: the
Hugging Face dataset [`materialyze/matcalc-bench`](https://huggingface.co/datasets/materialyze/matcalc-bench)
(Equilibrium, Elasticity, Softening), the packaged Phonon and Kappa datasets, and Matbench Discovery's WBM
files (Discovery) and dimer curves (Diatomics). The three molecular benchmarks (Noncovalent, Conformers,
Reactions) compare it with coupled-cluster energies of molecules, as MLIPAudit does.
Units follow ASE: energies in eV, forces in eV/Å, stresses in eV/Å³; moduli are reported in GPa; the
molecular benchmarks report energies in kcal/mol, as their references do.

The molecular benchmarks measure energy differences of a fraction of a kcal/mol to a few kcal/mol between
structures whose total energies are large when an MLIP's energies include the atomic energies of
all-electron quantum chemistry, as for MLIPs trained on molecular data (with MACE-OFF23, about 1.5·10⁴ eV for
a typical complex and 3.4·10⁴ eV for a typical conformer, up to 3.5·10⁵ eV). In float32 such a total energy
is a whole multiple of 0.01–0.09 kcal/mol for most of these structures (up to 0.36 kcal/mol for the largest
conformers and 0.72 kcal/mol for the heaviest complexes), and the order in which a GPU sums the atomic
energies moves it by several of these steps. **Evaluate the MLIP in float64 for the molecular benchmarks**;
results in float32, such as MLIPAudit's, carry this rounding ([validation.md](validation.md), section 10).

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

**Shards.** The whole set takes 3.4 h on one H100 MIG slice ([validation.md](validation.md), section 7). It
can run as n separate jobs instead: `DiscoveryBenchmark(shard=(k, n))` takes every n-th crystal of the draw
from the k-th on (k = 0, ..., n − 1; the draw of `n_samples` and `seed` is the same for all shards, and only
the shard's crystals are read). The shard is kept in the checkpoint, so that a checkpoint is not resumed as
another shard. `DiscoveryBenchmark.merge_shards` joins the tables of the n finished shards, given in shard
order, into the table of the whole run, and `DiscoveryBenchmark.metrics` computes the summary from that table
alone:

```python
tables = [DiscoveryBenchmark(shard=(k, 4)).run(model, "my-mlip") for k in range(4)]  # one k per job
table = DiscoveryBenchmark.merge_shards(tables)
metrics = DiscoveryBenchmark.metrics(table, "my-mlip")
```

`validation/tsubame_discovery_shards.sh` runs the four shards as a TSUBAME job array (one `gpu_h` slice each)
and `validation/merge_shards.py` joins their tables.

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

## Diatomics — `DiatomicsBenchmark` (`benchmarks/diatomics.py`)

Dataset: Matbench Discovery's PBE curves of the homonuclear dimers X2, X = H…U (`diatomics-dft.json.gz`,
Figshare file 68541277, CC BY 4.0, downloaded on first use): VASP with Materials Project settings in a 15 Å
box, 50 separations from 0.8 r_cov to 6 Å, at each separation the lowest of several constrained spin
states, with a narrow cleaning of likely SCF artifacts (documented with the file). The benchmark scores the
87 elements that the Materials Project covers (all but Po, At, Rn, Fr and Ra), as Matbench Discovery does.
Reference: J. Riebesell et al., Nat. Mach. Intell. 7, 836 (2025); the smoothness metrics come from the MACE-MP
paper (I. Batatia et al., arXiv:2401.00096) and MLIP Arena.

1. **Single points** of every dimer at 119 log-spaced separations from 0.1 to 6 Å, one atom at the centre of a
   50 Å box and the other along x (Matbench Discovery's grid); no relaxation.
2. **Metrics** of each element's curve (`properties/diatomics.py`), in its window from 0.9 r_cov to
   min(3.1 r_vdW, 6 Å):
   - smoothness: tortuosity (1 for a single well), the number of sign changes of the energy steps and of
     the force, the size of the steps at those changes (`energy_jump`, `force_jump`), and the total
     variation of the force;
   - against PBE: `pbe_energy_mae` (both curves shifted to zero at their largest separation) and
     `pbe_force_mae` on 200 common points, the errors of the bond length, the well depth and the harmonic
     vibrational wavenumber (cm⁻¹; quadratic fit of the five points around the minimum, for wells deeper
     than 0.05 eV), and `pbe_wall_dist_mae`, the error of the separation at 1, 5, 10, 20, 50 and 100 eV
     above the minimum, down to 0.8 r_cov.

A curve with a non-finite energy or force between 0.8 r_cov and the end of its window gets no metrics;
non-finite points below that range are left out. PBE curves too rough to score against (energy steps
adding up to 1.5 eV at three or more sign changes in the window: 8 lanthanides in the current file) give
their elements the smoothness metrics only.

Columns: per model, the twelve metrics and `status`. Summary: the mean of each metric over the elements that
have it, and `pbe_vib_freq_coverage`: how many of the elements whose PBE curve has a vibrational frequency
got a frequency error.

## Noncovalent — `NoncovalentBenchmark` (`benchmarks/noncovalent.py`)

Dataset: the dissociation curves of the Non-Covalent Interactions Atlas (J. Řezáč and co-workers,
[nciatlas.org](http://www.nciatlas.org), CC BY 4.0): the six packaged sets of
[Honza-R/NCIAtlas](https://github.com/Honza-R/NCIAtlas) at commit `1816bfc` (15 MB, MD5-checked, downloaded
on first use). 2,206 complexes of two molecules, neutral or ionic, each at 10 separations (the closest
contact scaled from 0.8 to 2.0 times its equilibrium length) or, for the repulsive contacts of R739x5, at 5
separations (scaled from 1.0 to 1.25), with CCSD(T)/CBS interaction energies at every point.

| Set | Complexes | Data set in the summary | Reference |
|---|---:|---|---|
| D442x10 | 442 | Dispersion | J. Řezáč, Phys. Chem. Chem. Phys. 24, 14780 (2022) |
| HB375x10 | 375 | Hydrogen bonds | J. Řezáč, J. Chem. Theory Comput. 16, 2355 (2020) |
| HB300SPXx10 | 300 | Hydrogen bonds | J. Řezáč, J. Chem. Theory Comput. 16, 6305 (2020) |
| IHB100x10 | 100 | Ionic hydrogen bonds | J. Řezáč, J. Chem. Theory Comput. 16, 2355 (2020) |
| R739x5 | 739 | Repulsive contacts | K. Kříž, M. Nováček, J. Řezáč, J. Chem. Theory Comput. 17, 1548 (2021) |
| SH250x10 | 250 | Sigma hole | K. Kříž, J. Řezáč, Phys. Chem. Chem. Phys. 24, 14794 (2022) |

The recipe and the metrics are those of the noncovalent-interactions benchmark of MLIPAudit (L. Wehrhan et
al., arXiv:2511.20487), whose published results the benchmark reproduces up to the rounding of MLIPAudit's
float32 energies ([validation.md](validation.md), section 10):

1. **Single points** of every point of every curve: the complex in a 50 Å periodic box
   (`structures.molecule_in_box`, at least 40 Å wider than the complex), its total charge in
   `Atoms.info["charge"]` and spin multiplicity 1 in `Atoms.info["spin"]`, where charge-aware ASE calculators
   read them (`TorchSimSimulator` passes them into the TorchSim state); no relaxation.
2. **Interaction energy** of the MLIP and of the reference (kcal/mol, `properties/molecules.py`): the lowest
   energy of the curve minus the energy at the largest separation; for the repulsive contacts of R739x5, the
   highest energy instead of the lowest.

`elements` lists the elements the model supports: complexes with other elements are skipped (status
`skipped: ...`), as in MLIPAudit (478 of the 2,206 complexes for the ten elements of MACE-OFF23). `sets`
selects data sets.

Columns: `interaction_energy_ref`, per model `interaction_energy` and `status`, and `dataset`, `group`
(MLIPAudit's names; the group HBCNO of D442x10 is split into HCNO and Boron, the complexes with boron) and
`name`. Summary: MAE and RMSE (kcal/mol) over all complexes, per data set and per subset
(`"<data set>: <group>"`), and the counts.

## Conformers — `ConformerBenchmark` (`benchmarks/conformers.py`)

Dataset: the conformer benchmark of D. L. Folmsbee and G. R. Hutchison, Int. J. Quantum Chem. 121, e26381 (2021)
([hutchisonlab/conformer-benchmark](https://github.com/hutchisonlab/conformer-benchmark), MIT license; the
repository at commit `0109c8e`, a 41 MB archive, MD5-checked, downloaded on first use): up to 10 conformers
of each of 702 drug-like molecules (86 of them ions, charges −1 to +2; elements H, C, N, O, F, P, S, Cl,
Br), optimized with B3LYP-D3BJ, with DLPNO-CCSD(T) single-point energies.

The recipe and the metrics are those of MLIPAudit's conformer-selection benchmark (compared with its
published results in [validation.md](validation.md), section 10):

1. The 693 molecules with at least three conformers (`MIN_CONFORMERS`).
2. **Single points** of their 6,745 conformers, each in a 50 Å periodic box with its charge in `Atoms.info`;
   no relaxation.
3. Per molecule, the energies relative to the conformer lowest in the reference (the zero of both the
   reference and the prediction): their mean absolute and root-mean-square errors (kcal/mol) and the
   Spearman rank correlation of the predicted with the reference energies (`properties/molecules.py`).

`elements` skips molecules with other elements, as for Noncovalent.

Columns: `energies_ref` and per model `energies` (the relative energies of the conformers in the order of
the data, kcal/mol), `mae`, `rmse`, `spearman`, `status`, and `charge`, `n_conformers`. Summary: the means of
`mae`, `rmse` and `spearman` over the molecules (MLIPAudit's `avg_mae`, `avg_rmse`) and the counts.

## Reactions — `ReactionBenchmark` (`benchmarks/reactions.py`)

Dataset: RDB7 (K. A. Spiekermann, L. Pattanaik, W. H. Green, Sci. Data 9, 417 (2022);
[Zenodo record 6618262](https://zenodo.org/records/6618262), CC BY 4.0), the refined reactions of C. A.
Grambow, L. Pattanaik, W. H. Green, Sci. Data 7, 137 (2020): 11,926 elementary reactions of closed-shell
molecules with up to 7 heavy atoms (H, C, N, O; 4 to 23 atoms), found by transition-state searches from
GDB-7 reactants. Reactants, transition states and products are optimized with ωB97X-D3/def2-TZVP (the 3,534
reactions that break the reactant apart have 2 or 3 product molecules, each optimized on its own), and their
energies computed with CCSD(T)-F12a/cc-pVDZ-F12. The benchmark reads geometries and energies from the Molpro
outputs of these single points (`ccsdtf12_dz.tar.gz`, 143 MB, MD5-checked, downloaded on first use).

The recipe and the metrics are those of MLIPAudit's reactivity benchmark, which uses the same reactants and
transition states with the ωB97X-D3 energies of Grambow et al. and their product structures
([validation.md](validation.md), section 11):

1. **Single points** of the reactant, the transition state and each product molecule, every one in a 50 Å
   periodic box (all neutral singlets); no relaxation.
2. **Barrier height** (transition state minus reactant) and **reaction energy** (the products minus the
   reactant) of the MLIP and of the reference, electronic energies in kcal/mol (`properties/molecules.py`;
   the barrier heights in RDB7's tables add zero-point energies of ωB97X-D3 frequencies, which an MLIP does
   not predict).

`elements` skips reactions with other elements, as for the other molecular benchmarks.

Columns: `barrier_ref`, `reaction_energy_ref`, per model `barrier`, `reaction_energy` and `status`, and
`n_products`. Summary: MAE, RMSE and mean signed error ME (prediction minus reference, kcal/mol) of the
barrier heights and of the reaction energies, and the counts.
