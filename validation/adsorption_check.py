"""Checks of the Adsorption benchmark (docs/validation.md, section 13).

python adsorption_check.py references
    The 39 ADS41 references taken from CE39 against Wellendorff et al.'s Tables 4a and 4b (experimental reaction
    energy minus the PBE zero-point energy change, kJ/mol), and the Surf13 references against the enthalpies
    kept in the dataset (H_ads - Delta H).
python adsorption_check.py dft <table.csv> [--model mace]
    A run (run_one.py) against DFT energies of the same reactions: PBE of Sharada et al. (2019) Table II for
    ADS41, and PBE+D3 of Araujo et al. (2022) Table 3 for its 25 molecular adsorptions of ADS41.
python adsorption_check.py parity [--model medium-omat-0] [--reactions ADS41-02,...] [--device cpu]
    ASE vs TorchSim (float64) on the same reactions: reaction energies, displacements, the FIRE steps of every
    relaxation and the status.
python adsorption_check.py d3
    torch-dftd's D3(BJ) of PBE (ASE) vs TorchSim's D3DispersionModel with torch-dftd's reference parameters:
    single points on adsorbed slabs and molecules of the dataset, built on the experimental lattices; and the D3
    term alone of a few reaction energies with the damping parameters of PBE and of RPBE.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KJ_PER_EV = 96.485332
KCAL_PER_EV = 23.060548

CE39 = {  # Wellendorff et al., Surf. Sci. 640, 36 (2015), Tables 4a and 4b: Delta E_exp - Delta ZPE (kJ/mol)
    1: -124, 2: -124, 3: -144, 4: -157, 5: -142, 6: -164, 7: -57, 8: -161, 9: -119, 10: -299, 11: -119, 12: -182,
    13: -163, 14: -485, 15: -530, 16: -208, 17: -355, 18: -72, 19: -100, 20: -87, 21: -72, 22: -90, 23: -313,
    24: -455, 25: -209, 26: -60, 27: -84, 28: -55, 29: -14, 30: -27, 31: -39, 32: -48, 33: -162, 34: -66, 35: -61,
    36: -70, 37: -123, 38: -55, 39: -66,
}

PBE = [  # Sharada et al., Phys. Rev. B 100, 035439 (2019), Table II, PBE (eV), in the order of ADS41-01 ... 41
    -1.65, -3.85, -1.66, -1.63, -0.75, -1.93, -1.81, -1.86, -1.95, -1.64, -1.84, -1.88, -1.01, -1.10, -1.20, -0.96,
    -1.07, -2.84, -4.37, -2.05, -2.23, -1.83, -5.07, -4.58, -2.37, -4.33, -0.04, -0.05, -0.07, -0.05, -0.05, -0.05,
    -0.98, -0.63, -1.84, -0.25, -0.21, -0.02, -0.66, -0.21, -0.43,
]

PBE_D3 = {  # Araujo et al., Nat. Commun. 13, 6853 (2022), Table 3, PBE+D3 (kcal/mol), molecular adsorption
    "ADS41-04": -46.6, "ADS41-05": -23.0, "ADS41-06": -51.2, "ADS41-07": -49.4, "ADS41-08": -50.2,
    "ADS41-09": -51.2, "ADS41-10": -42.9, "ADS41-11": -49.5, "ADS41-12": -50.7, "ADS41-20": -56.0,
    "ADS41-21": -58.1, "ADS41-22": -48.1, "ADS41-27": -10.4, "ADS41-28": -15.0, "ADS41-29": -20.1,
    "ADS41-30": -19.4, "ADS41-31": -20.5, "ADS41-32": -22.4, "ADS41-33": -52.3, "ADS41-34": -43.7,
    "ADS41-36": -18.4, "ADS41-37": -14.1, "ADS41-38": -6.0, "ADS41-40": -16.8, "ADS41-41": -15.7,
}

PARITY_REACTIONS = (  # on fcc(111), fcc(100), five-layer Pt(111), MgO(001), rutile and anatase slabs
    "ADS41-02,ADS41-10,ADS41-16,ADS41-26,ADS41-30,ADS41-33,ADS41-40,Surf13-03,Surf13-04,Surf13-09,Surf13-12"
)


def references() -> None:
    """The references of the dataset against the published tables they come from."""
    from matcalc.benchmarks.adsorption import DATASET

    reactions = json.loads(DATASET.read_text())["reactions"]
    worst, seen = 0.0, set()
    for reaction in reactions:
        reference = reaction["reference"]
        if reaction["subset"] == "ADS41":
            match = re.search(r"CE39 reaction (\d+)", reference["source"])
            if match is None:
                print(f"{reaction['id']}  {reaction['equation']:<58s} {reference['energy']:+.2f} (not in CE39)")
                continue
            number = int(match.group(1))
            seen.add(number)
            published = CE39[number] / KJ_PER_EV
        else:
            published = reference["enthalpy"] - reference["vibrational_enthalpy"]
        difference = reference["energy"] - published
        worst = max(worst, abs(difference))
        print(
            f"{reaction['id']}  {reaction['equation']:<58s} {reference['energy']:+.3f} {published:+.3f} "
            f"{difference:+.4f}"
        )
    print(f"largest difference {worst:.4f} eV; CE39 reactions covered: {len(seen)} of {len(CE39)}")


def dft(table_file: str, model: str) -> None:
    """A run against PBE (Sharada et al. Table II) and PBE+D3 (Araujo et al. Table 3)."""
    table = pd.read_csv(table_file).set_index("reaction")
    energy, displacement = table[f"energy_{model}"], table[f"displacement_{model}"]
    ads41 = table[table["subset"] == "ADS41"].index
    rows = pd.DataFrame(
        {
            "formula": table.loc[ads41, "formula"],
            "category": table.loc[ads41, "category"],
            "exp": table.loc[ads41, "energy_exp"],
            "PBE": PBE,
            "PBE+D3": [PBE_D3[r] / KCAL_PER_EV if r in PBE_D3 else np.nan for r in ads41],
            "MLIP": energy[ads41],
            "displacement": displacement[ads41],
        }
    )
    rows["MLIP - PBE"] = rows["MLIP"] - rows["PBE"]
    rows["MLIP - PBE+D3"] = rows["MLIP"] - rows["PBE+D3"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(rows.round(2).to_string())
    chemisorption = rows["category"] == "chemisorption"
    everything = rows["category"].notna()
    for label, mask in (("chemisorption", chemisorption), ("dispersion", ~chemisorption), ("all", everything)):
        to_pbe = rows["MLIP - PBE"][mask].abs()
        print(
            f"{label:<14s} MAE (eV): MLIP - exp {(rows['MLIP'] - rows['exp'])[mask].abs().mean():.3f}, "
            f"PBE - exp {(rows['PBE'] - rows['exp'])[mask].abs().mean():.3f}, "
            f"MLIP - PBE {to_pbe.mean():.3f} (median {to_pbe.median():.3f})"
        )
    d3 = rows.dropna(subset=["PBE+D3"])
    to_d3 = d3["MLIP - PBE+D3"]
    print(
        f"{len(d3)} molecular adsorptions, MAE (eV): MLIP - PBE+D3 {to_d3.abs().mean():.3f} "
        f"(median {to_d3.abs().median():.3f}, ME {to_d3.mean():+.3f}), "
        f"PBE+D3 - exp {(d3['PBE+D3'] - d3['exp']).abs().mean():.3f}, "
        f"MLIP - exp {(d3['MLIP'] - d3['exp']).abs().mean():.3f}"
    )


class _Steps:
    """A simulator that records the number of steps and the energy of every relaxation."""

    def __init__(self, inner: object) -> None:
        self.inner, self.steps = inner, []

    @property
    def batched(self) -> bool:
        return getattr(self.inner, "batched", False)

    def relax(self, structures: list, **kwargs: object) -> list:
        results = self.inner.relax(structures, **kwargs)
        self.steps += [(len(s), r.n_steps, r.energy) for s, r in zip(structures, results, strict=True)]
        return results

    def single_point(self, structures: list, **kwargs: object) -> list:
        return self.inner.single_point(structures, **kwargs)


def parity(model: str, reaction_ids: str, device: str | None) -> None:
    """ASE vs TorchSim (float64) on the same reactions."""
    import matcalc
    from matcalc.simulation import TorchSimSimulator

    sys.path.insert(0, str(Path(__file__).parent))
    from mace_models import load_mace

    ids = reaction_ids.split(",")
    runs = {}
    for backend in ("ase", "torchsim"):
        potential = load_mace(backend, model=model, dtype="float64", device=device)
        simulator = _Steps(
            TorchSimSimulator(potential, show_progress=False)
            if backend == "torchsim"
            else matcalc.ASESimulator(potential, show_progress=False)
        )
        bench = matcalc.AdsorptionBenchmark()
        bench.materials = [m for m in bench.materials if m.material_id in ids]
        start = time.perf_counter()
        table = bench.run(simulator, "mlip")
        runs[backend] = (table, simulator.steps, time.perf_counter() - start, dict(bench.timings))
    (ase, ase_steps, ase_time, ase_timings), (ts, ts_steps, ts_time, ts_timings) = runs["ase"], runs["torchsim"]
    atoms = np.array([n for n, _, _ in ase_steps])
    print(
        json.dumps(
            {
                "reactions": len(ase),
                "max |dE| reaction (eV)": float(np.abs(ase["energy_mlip"] - ts["energy_mlip"]).max()),
                "max |d displacement| (A)": float(np.abs(ase["displacement_mlip"] - ts["displacement_mlip"]).max()),
                "same status": bool((ase["status_mlip"] == ts["status_mlip"]).all()),
                "relaxations": len(ase_steps),
                "relaxations with other step counts": sum(
                    a[1] != t[1] for a, t in zip(ase_steps, ts_steps, strict=True)
                ),
                "max |dE|/atom (eV)": float(
                    np.max(np.abs(np.array([e for *_, e in ase_steps]) - np.array([e for *_, e in ts_steps])) / atoms)
                ),
                "time (s)": {"ase": round(ase_time, 1), "torchsim": round(ts_time, 1)},
                "stages (s)": {"ase": ase_timings, "torchsim": ts_timings},
            },
            indent=1,
            default=lambda x: round(x, 1),
        )
    )


def d3() -> None:
    """torch-dftd's D3 vs TorchSim's D3DispersionModel, single points on structures of the dataset."""
    import torch
    import torch_sim as ts
    from ase.units import Bohr

    import matcalc
    from matcalc.surfaces import add_adsorbates, build_slab

    sys.path.insert(0, str(Path(__file__).parent))
    from mace_models import torchsim_d3
    from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

    bench = matcalc.AdsorptionBenchmark()
    definitions = bench.definitions
    structures = {}
    adsorbed = (
        "CO/Pt(111) ontop", "C6H6/Pt(111) bridge", "C10H8/Pt(111) ontop", "CH4/MgO(001)", "H2O/TiO2 anatase(101)",
    )
    for name in adsorbed:
        entry = definitions["adsorbed"][name]
        slab = definitions["slabs"][entry["slab"]]
        crystal = bench._crystal(slab["crystal"])  # noqa: SLF001 - the experimental lattice of the dataset
        structures[name] = add_adsorbates(build_slab(crystal, slab), entry["adsorbates"])
    for name in ("CO", "C6H6"):
        structures[name] = bench._molecule(name)  # noqa: SLF001
    cpu = torch.device("cpu")
    dispersion = torchsim_d3("pbe", device="cpu", dtype="float64")
    for cnthr in (40.0, 95.0):
        ase_d3 = TorchDFTD3Calculator(device="cpu", damping="bj", xc="pbe", dtype=torch.float64, cnthr=cnthr * Bohr)
        print(f"torch-dftd with coordination numbers out to {cnthr:.0f} Bohr:")
        for name, structure in structures.items():
            atoms = structure.copy()
            atoms.set_constraint()
            atoms.calc = ase_d3
            energy, forces = atoms.get_potential_energy(), atoms.get_forces()
            out = dispersion(ts.io.atoms_to_state([atoms], device=cpu, dtype=torch.float64))
            print(
                f"  {name:<24s} {len(atoms):4d} atoms  E {energy:+.6f} eV  TorchSim - torch-dftd "
                f"{float(out['energy'][0]) - energy:+.1e} eV, "
                f"forces {np.abs(out['forces'].numpy() - forces).max():.1e} eV/A"
            )
    damping_parameters(bench)


def damping_parameters(bench: object) -> None:
    """The D3 term alone of a few reaction energies, with the damping parameters of PBE and of RPBE, on the
    starting geometries of the dataset (experimental lattices).
    """
    from mace_models import ase_d3

    from matcalc.properties.adsorption import reaction_energy
    from matcalc.surfaces import add_adsorbates, build_slab

    definitions = bench.definitions  # type: ignore[attr-defined]
    reactions = {r["id"]: r for r in definitions["reactions"]}

    def structure(name: str) -> object:
        if name in definitions["molecules"]:
            return bench._molecule(name)  # type: ignore[attr-defined]  # noqa: SLF001
        entry = definitions["adsorbed"].get(name)
        slab = definitions["slabs"][entry["slab"] if entry else name]
        atoms = build_slab(bench._crystal(slab["crystal"]), slab)  # type: ignore[attr-defined]  # noqa: SLF001
        return add_adsorbates(atoms, entry["adsorbates"]) if entry else atoms

    print("D3 alone in the reaction energy (eV), damping parameters of PBE / RPBE:")
    calculators = {xc: ase_d3(xc, device="cpu", dtype="float64") for xc in ("pbe", "rpbe")}
    for reaction_id in ("ADS41-18", "ADS41-03", "ADS41-35", "ADS41-32", "ADS41-10", "Surf13-01"):
        terms = reactions[reaction_id]["terms"]
        values = []
        for calculator in calculators.values():
            energies = {}
            for name in terms:
                atoms = structure(name)
                atoms.set_constraint()
                atoms.calc = calculator
                energies[name] = atoms.get_potential_energy()
            values.append(reaction_energy(terms, energies))
        print(f"  {reaction_id}  {reactions[reaction_id]['equation']:<48s} {values[0]:+.2f} / {values[1]:+.2f}")


def main() -> None:
    """Run the check named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("references")
    dft_parser = commands.add_parser("dft")
    dft_parser.add_argument("table")
    dft_parser.add_argument("--model", default="mace")
    parity_parser = commands.add_parser("parity")
    parity_parser.add_argument("--model", default="medium-omat-0")
    parity_parser.add_argument("--reactions", default=PARITY_REACTIONS)
    parity_parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    commands.add_parser("d3")
    args = parser.parse_args()
    if args.command == "references":
        references()
    elif args.command == "dft":
        dft(args.table, args.model)
    elif args.command == "parity":
        parity(args.model, args.reactions, args.device)
    else:
        d3()


if __name__ == "__main__":
    main()
