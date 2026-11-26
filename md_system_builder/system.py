"""A plain numpy description of a molecular system, and its RDKit round trip.

Every tab works on :class:`System`: elements, Cartesian positions, bonds, formal
charges and per-atom residue data (molecule index, residue name and number, atom
name, chain).  Converting to an RDKit molecule happens once, at the end of an
operation, so packing tens of thousands of atoms never touches RDKit per atom
more than once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cellmeta import PeriodicCell, read_cell, write_cell
from .elements import atomic_mass

# Residue names this plugin gives to molecules without one.
DEFAULT_RESNAME = "MOL"


@dataclass
class System:
    elements: list[str] = field(default_factory=list)
    xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    bonds: list[tuple[int, int, float]] = field(default_factory=list)  # (i, j, order)
    charges: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    mol_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    resnames: list[str] = field(default_factory=list)
    resnums: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    atom_names: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    cell: PeriodicCell | None = None

    def __post_init__(self):
        n = len(self.elements)
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(n, 3)
        self.charges = _fill(self.charges, n, 0, int)
        self.mol_index = _fill(self.mol_index, n, 0, int)
        self.resnums = _fill(self.resnums, n, 1, int)
        if len(self.resnames) != n:
            self.resnames = [DEFAULT_RESNAME] * n
        if len(self.atom_names) != n:
            self.atom_names = default_atom_names(self.elements)
        if len(self.chains) != n:
            self.chains = ["A"] * n

    # -- basics -----------------------------------------------------------

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    @property
    def n_molecules(self) -> int:
        return int(self.mol_index.max()) + 1 if self.n_atoms else 0

    def total_mass(self) -> float:
        return float(sum(atomic_mass(e) for e in self.elements))

    def net_charge(self) -> int:
        return int(self.charges.sum())

    def molecules(self) -> list[np.ndarray]:
        """Atom indices of each molecule, in molecule order."""
        if not self.n_atoms:
            return []
        order = np.argsort(self.mol_index, kind="stable")
        ids = self.mol_index[order]
        cuts = np.flatnonzero(np.diff(ids)) + 1
        return list(np.split(order, cuts))

    def copy(self) -> System:
        return System(
            list(self.elements),
            self.xyz.copy(),
            list(self.bonds),
            self.charges.copy(),
            self.mol_index.copy(),
            list(self.resnames),
            self.resnums.copy(),
            list(self.atom_names),
            list(self.chains),
            None if self.cell is None else self.cell.copy(),
        )

    def subset(self, keep) -> System:
        """The atoms in ``keep`` (sorted), with bonds and molecule indices renumbered."""
        keep = np.asarray(sorted(int(i) for i in keep), dtype=int)
        remap = -np.ones(self.n_atoms, dtype=int)
        remap[keep] = np.arange(len(keep))
        bonds = [
            (int(remap[i]), int(remap[j]), o)
            for i, j, o in self.bonds
            if remap[i] >= 0 and remap[j] >= 0
        ]
        _, mol_index = np.unique(self.mol_index[keep], return_inverse=True)
        return System(
            [self.elements[i] for i in keep],
            self.xyz[keep],
            bonds,
            self.charges[keep],
            mol_index.astype(int),
            [self.resnames[i] for i in keep],
            self.resnums[keep],
            [self.atom_names[i] for i in keep],
            [self.chains[i] for i in keep],
            None if self.cell is None else self.cell.copy(),
        )

    def residue_counts(self) -> list[tuple[str, int]]:
        """(residue name, molecule count) for consecutive runs of identical molecules."""
        runs: list[tuple[str, int]] = []
        for atoms in self.molecules():
            name = self.resnames[int(atoms[0])]
            if runs and runs[-1][0] == name:
                runs[-1] = (name, runs[-1][1] + 1)
            else:
                runs.append((name, 1))
        return runs


def _fill(values, n, default, dtype):
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if len(arr) != n:
        arr = np.full(n, default, dtype=dtype)
    return arr


def default_atom_names(elements) -> list[str]:
    counts = {}
    names = []
    for el in elements:
        counts[el] = counts.get(el, 0) + 1
        names.append(f"{el}{counts[el]}"[:4])
    return names


def concatenate(parts: list[System], cell: PeriodicCell | None = None) -> System:
    """Join systems in order; molecule indices and residue numbers continue."""
    elements, xyz, bonds, charges, mol_index = [], [], [], [], []
    resnames, resnums, names, chains = [], [], [], []
    atom_offset = 0
    mol_offset = 0
    res_offset = 0
    for part in parts:
        if part.n_atoms == 0:
            continue
        elements.extend(part.elements)
        xyz.append(part.xyz)
        bonds.extend((i + atom_offset, j + atom_offset, o) for i, j, o in part.bonds)
        charges.append(part.charges)
        mol_index.append(part.mol_index + mol_offset)
        resnames.extend(part.resnames)
        # Keep each part's own numbering pattern but never overlap the previous part.
        resnums.append(part.resnums - part.resnums.min() + 1 + res_offset)
        names.extend(part.atom_names)
        chains.extend(part.chains)
        atom_offset += part.n_atoms
        mol_offset += part.n_molecules
        res_offset = int(resnums[-1].max())
    if not elements:
        return System(cell=cell)
    return System(
        elements,
        np.vstack(xyz),
        bonds,
        np.concatenate(charges),
        np.concatenate(mol_index),
        resnames,
        np.concatenate(resnums),
        names,
        chains,
        cell,
    )


def perceive_molecules(n_atoms: int, bonds) -> np.ndarray:
    """Connected components of the bond graph; molecule index per atom, in atom order."""
    parent = list(range(n_atoms))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in bonds:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n_atoms)], dtype=int)
    _, first = np.unique(roots, return_index=True)
    order = {int(roots[k]): rank for rank, k in enumerate(sorted(first))}
    return np.array([order[int(r)] for r in roots], dtype=int)


# ---------------------------------------------------------------------------
# RDKit round trip
# ---------------------------------------------------------------------------


def from_rdkit(mol, resname: str | None = None, conf_id: int = -1) -> System:
    """System from an RDKit molecule (needs a conformer).

    Residue data comes from the atoms' PDB residue info when present; otherwise each
    connected fragment is one molecule named ``resname`` (or ``MOL``).
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("The molecule has no atoms.")
    if mol.GetNumConformers() == 0:
        raise ValueError("The molecule has no 3D coordinates. Generate 3D first.")
    conf = mol.GetConformer(conf_id)
    n = mol.GetNumAtoms()
    xyz = np.array(conf.GetPositions(), dtype=float).reshape(n, 3)
    elements = []
    charges = np.zeros(n, dtype=int)
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if atom.HasProp("custom_symbol"):
            custom = atom.GetProp("custom_symbol").strip()
            if custom:
                symbol = custom
        elements.append(symbol)
        charges[atom.GetIdx()] = atom.GetFormalCharge()
    bonds = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), float(b.GetBondTypeAsDouble()))
        for b in mol.GetBonds()
    ]
    mol_index = perceive_molecules(n, bonds)

    resnames, resnums, names, chains = [], [], [], []
    have_info = True
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            have_info = False
            break
        resnames.append(
            (info.GetResidueName() or DEFAULT_RESNAME).strip() or DEFAULT_RESNAME
        )
        resnums.append(int(info.GetResidueNumber()))
        names.append((info.GetName() or "").strip())
        chains.append((info.GetChainId() or "A").strip() or "A")
    if not have_info:
        name = (resname or DEFAULT_RESNAME)[:4]
        resnames = [name] * n
        resnums = list(mol_index + 1)
        names = default_atom_names(elements)
        chains = ["A"] * n
    else:
        fixed = default_atom_names(elements)
        names = [nm or fixed[k] for k, nm in enumerate(names)]
        if resname:
            resnames = [resname[:4]] * n

    system = System(
        elements,
        xyz,
        bonds,
        charges,
        mol_index,
        resnames,
        np.asarray(resnums, dtype=int),
        names,
        chains,
        read_cell(mol),
    )
    return system


_BOND_TYPES = None


def _bond_type(order: float):
    from rdkit import Chem

    global _BOND_TYPES
    if _BOND_TYPES is None:
        _BOND_TYPES = {
            1.0: Chem.BondType.SINGLE,
            1.5: Chem.BondType.AROMATIC,
            2.0: Chem.BondType.DOUBLE,
            3.0: Chem.BondType.TRIPLE,
        }
    return _BOND_TYPES.get(float(order), Chem.BondType.SINGLE)


def to_rdkit(system: System):
    """RDKit molecule with conformer, PDB residue info and the ``moleditpy_cell`` property."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for k, el in enumerate(system.elements):
        symbol = el
        atom = Chem.Atom(symbol) if _is_rdkit_element(symbol) else Chem.Atom(0)
        atom.SetFormalCharge(int(system.charges[k]))
        atom.SetNoImplicit(True)
        info = Chem.AtomPDBResidueInfo()
        info.SetName(_pdb_atom_name(system.atom_names[k], el))
        info.SetResidueName(system.resnames[k][:4])
        info.SetResidueNumber(int(system.resnums[k]))
        info.SetChainId(system.chains[k][:1] or "A")
        info.SetSerialNumber(k + 1)
        info.SetIsHeteroAtom(system.resnames[k] not in _AMINO_ACIDS)
        atom.SetMonomerInfo(info)
        rw.AddAtom(atom)
    for i, j, order in system.bonds:
        if rw.GetBondBetweenAtoms(int(i), int(j)) is None:
            rw.AddBond(int(i), int(j), _bond_type(order))
    mol = rw.GetMol()
    for i, j, order in system.bonds:
        if float(order) == 1.5:
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            bond.SetIsAromatic(True)
            mol.GetAtomWithIdx(int(i)).SetIsAromatic(True)
            mol.GetAtomWithIdx(int(j)).SetIsAromatic(True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - keep an unsanitized but complete molecule
        mol.UpdatePropertyCache(strict=False)
    conf = Chem.Conformer(system.n_atoms)
    conf.Set3D(True)
    for k, (x, y, z) in enumerate(system.xyz):
        conf.SetAtomPosition(k, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    write_cell(mol, system.cell)
    return mol


_AMINO_ACIDS = {
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
    "ACE",
    "NME",
    "ASH",
    "GLH",
    "LYN",
    "CYM",
    "HYP",
    "NHE",
}


def _is_rdkit_element(symbol: str) -> bool:
    from .elements import is_known

    return is_known(symbol)


def _pdb_atom_name(name: str, element: str) -> str:
    """PDB columns 13-16: one-letter elements start in column 14."""
    name = (name or element)[:4]
    if len(name) < 4 and len(element) == 1:
        return f" {name:<3}"
    return f"{name:<4}"
