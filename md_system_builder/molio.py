"""Component input (SMILES, MOL/SDF/XYZ/PDB) and system output (PDB/GRO/XYZ)."""

from __future__ import annotations

import logging
import os

import numpy as np

from .cellmeta import PeriodicCell
from .system import System, from_rdkit

FILE_FILTER = "Molecule files (*.mol *.sdf *.xyz *.pdb);;All files (*)"
OUTPUT_FORMATS = ("pdb", "gro", "xyz")

logger = logging.getLogger(__name__)


def from_smiles(smiles: str, name: str = "MOL", seed: int = 42) -> System:
    """3D template from SMILES: RDKit ETKDG embedding plus an MMFF (or UFF) cleanup."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    smiles = (smiles or "").strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse the SMILES {smiles!r}.")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol, params) != 0:
            raise ValueError(f"RDKit could not generate 3D coordinates for {smiles!r}.")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        logger.debug("Ignored error", exc_info=True)
    return from_rdkit(mol, resname=name)


def from_file(path: str, name: str = "") -> System:
    """Template from a MOL/SDF (first record), XYZ (bonds perceived) or PDB file."""
    from rdkit import Chem

    if not os.path.isfile(path):
        raise ValueError(f"File not found:\n{path}")
    ext = os.path.splitext(path)[1].lower()
    mol = None
    if ext in (".mol", ".sdf"):
        supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
        mol = next((m for m in supplier if m is not None), None)
    elif ext == ".pdb":
        mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
    elif ext == ".xyz":
        mol = Chem.MolFromXYZFile(path)
        if mol is not None:
            from rdkit.Chem import rdDetermineBonds

            try:
                rdDetermineBonds.DetermineConnectivity(mol)
            except Exception:
                logger.debug("Ignored error", exc_info=True)
    else:
        raise ValueError(f"Unsupported file type {ext or '(none)'}.")
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"No molecule could be read from\n{path}")
    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        logger.debug("Ignored error", exc_info=True)
    default = os.path.splitext(os.path.basename(path))[0][:3].upper() or "MOL"
    return from_rdkit(mol, resname=name or (None if ext == ".pdb" else default))


# ---------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------


def write_pdb(system: System, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(pdb_text(system))


def pdb_text(system: System) -> str:
    lines = ["REMARK   1 Written by the MoleditPy MD System Builder plugin"]
    if system.cell is not None:
        (a, b, c), (al, be, ga) = system.cell.parameters
        lines.append(
            f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{al:7.2f}{be:7.2f}{ga:7.2f} P 1           1"
        )
    xyz = _pdb_frame(system)
    for k in range(system.n_atoms):
        el = system.elements[k]
        name = system.atom_names[k][:4]
        name = f" {name:<3}" if len(name) < 4 and len(el) == 1 else f"{name:<4}"
        record = "ATOM  " if system.resnames[k] in _PROTEIN else "HETATM"
        serial = (k + 1) % 100000
        resnum = int(system.resnums[k]) % 10000
        charge = int(system.charges[k])
        charge_text = f"{abs(charge)}{'+' if charge > 0 else '-'}" if charge else "  "
        x, y, z = xyz[k]
        lines.append(
            f"{record}{serial:5d} {name}{' '}{system.resnames[k][:3]:>3} "
            f"{system.chains[k][:1] or 'A'}{resnum:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}          "
            f"{el.upper():>2}{charge_text}"
        )
    if system.n_atoms < 100000:
        for i, j in _conect_pairs(system):
            lines.append(f"CONECT{i + 1:5d}{j + 1:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def _pdb_frame(system: System) -> np.ndarray:
    """PDB CRYST1 implies the standard orientation (a on x, b in the xy plane) with the
    origin at 0; rotate and shift so a triclinic cell matches its CRYST1 record."""
    if system.cell is None:
        return system.xyz
    cell = system.cell
    frac = cell.to_fractional(system.xyz)
    standard = PeriodicCell.from_parameters(*cell.parameters)
    return frac @ standard.lattice


def _conect_pairs(system: System):
    seen = set()
    for i, j, _ in system.bonds:
        if system.resnames[int(i)] in _PROTEIN:
            continue
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key not in seen:
            seen.add(key)
            yield key


def write_gro(
    system: System, path: str, title: str = "MoleditPy MD System Builder"
) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(gro_text(system, title))


def gro_text(system: System, title: str = "MoleditPy MD System Builder") -> str:
    """GROMACS .gro (nm).  Triclinic boxes are written in GROMACS' lower-triangular form."""
    lines = [title, f"{system.n_atoms:5d}"]
    xyz = system.xyz
    box_line = "   0.00000   0.00000   0.00000"
    if system.cell is not None:
        cell = system.cell
        standard = PeriodicCell.from_parameters(*cell.parameters)
        xyz = cell.to_fractional(system.xyz) @ standard.lattice
        a, b, c = standard.lattice / 10.0
        if cell.is_orthorhombic() or np.allclose(
            [a[1], a[2], b[2], b[0], c[0], c[1]], 0, atol=1e-6
        ):
            box_line = f"{a[0]:10.5f}{b[1]:10.5f}{c[2]:10.5f}"
        else:
            box_line = (
                f"{a[0]:10.5f}{b[1]:10.5f}{c[2]:10.5f}{a[1]:10.5f}{a[2]:10.5f}"
                f"{b[0]:10.5f}{b[2]:10.5f}{c[0]:10.5f}{c[1]:10.5f}"
            )
    for k in range(system.n_atoms):
        x, y, z = xyz[k] / 10.0
        lines.append(
            f"{int(system.resnums[k]) % 100000:5d}{system.resnames[k][:5]:<5}"
            f"{system.atom_names[k][:5]:>5}{(k + 1) % 100000:5d}{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    lines.append(box_line)
    return "\n".join(lines) + "\n"


def write_xyz(system: System, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(xyz_text(system))


def xyz_text(system: System) -> str:
    """Extended XYZ: the comment line carries ``Lattice="..." pbc="T T T"``."""
    comment = "MoleditPy MD System Builder"
    if system.cell is not None:
        lat = " ".join(f"{v:.6f}" for v in system.cell.lattice.reshape(-1))
        pbc = " ".join("T" if p else "F" for p in system.cell.pbc)
        origin = " ".join(f"{v:.6f}" for v in system.cell.origin)
        comment = f'Lattice="{lat}" Origin="{origin}" pbc="{pbc}" Properties=species:S:1:pos:R:3'
    lines = [str(system.n_atoms), comment]
    for el, (x, y, z) in zip(system.elements, system.xyz):
        lines.append(f"{el:<2} {x:14.6f} {y:14.6f} {z:14.6f}")
    return "\n".join(lines) + "\n"


def write_system(system: System, path: str) -> None:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext == "pdb":
        write_pdb(system, path)
    elif ext == "gro":
        write_gro(system, path)
    elif ext == "xyz":
        write_xyz(system, path)
    else:
        raise ValueError(f"Unsupported output format .{ext}")


_PROTEIN = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "HID",
    "HIE",
    "HIP",
    "CYX",
}
