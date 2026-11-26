"""Prepare a protein for MD: hydrogens at pH 7, chain ends, disulfides, missing atoms.

``prepare(system)`` rebuilds every standard amino-acid residue from its Amber ff14SB
template (protein_templates.py):

- the heavy atoms of the input are kept (matched by their PDB names) and every
  hydrogen is rebuilt from the template, so names follow Amber and the protonation
  state is consistent: Lys and Arg charged, Asp and Glu charged, His neutral (HIE by
  default; HID or HIP when the input's hydrogens or residue name say so);
- chain ends are charged (NH3+ and COO-), with OXT added when it is missing; a chain
  break (C-N gap) starts a new chain end;
- cysteines whose sulfur atoms are within 2.5 A become CYX with an S-S bond;
- heavy atoms missing from a residue are added in the template's local geometry and
  reported (check them: the side chain conformation is the template's);
- crystal waters can be removed, and so can other groups (ions, ligands, additives);
  kept groups stay as they are.

Each new atom is placed from its bonded neighbour and two more placed atoms: the
template's three points are fitted onto the actual ones and the template position of
the new atom carried over, so bond lengths, angles and (for rotatable hydrogens) the
template's staggered torsions are reproduced. The amide hydrogen lies on the bisector of
the C(i-1)-N-CA angle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .protein_templates import TEMPLATES
from .system import System, perceive_molecules

#: Residue names read as amino acids, with their default ff14SB residue.
BASE = {
    "ALA": "ALA", "ARG": "ARG", "ASN": "ASN", "ASP": "ASP", "ASH": "ASH",
    "CYS": "CYS", "CYX": "CYS", "CYM": "CYM", "GLN": "GLN", "GLU": "GLU", "GLH": "GLH",
    "GLY": "GLY", "HIS": "HIS", "HID": "HID", "HIE": "HIE", "HIP": "HIP", "HSD": "HID",
    "HSE": "HIE", "HSP": "HIP", "HYP": "HYP", "ILE": "ILE", "LEU": "LEU", "LYS": "LYS",
    "LYN": "LYN", "MET": "MET", "PHE": "PHE", "PRO": "PRO", "SER": "SER", "THR": "THR",
    "TRP": "TRP", "TYR": "TYR", "VAL": "VAL", "ACE": "ACE", "NME": "NME", "NHE": "NHE",
}  # fmt: skip
CAPS = {"ACE", "NME", "NHE"}
WATER_NAMES = {"HOH", "WAT", "SOL", "H2O", "TIP3", "TIP", "DOD", "T3P"}
HISTIDINES = ("HIE", "HID", "HIP")

#: Atom names of other conventions -> Amber names (any residue / one residue).
ALIASES = {"OT1": "O", "OT2": "OXT", "OC1": "O", "OC2": "OXT", "O1": "O", "O2": "OXT"}
RESIDUE_ALIASES = {"ILE": {"CD": "CD1"}}

#: Double bonds of each residue (besides the backbone C=O) for a sensible 2D/3D display.
DOUBLE_BONDS = {
    "ARG": [("CZ", "NH2")],
    "ASN": [("CG", "OD1")],
    "ASP": [("CG", "OD1")],
    "ASH": [("CG", "OD1")],
    "GLN": [("CD", "OE1")],
    "GLU": [("CD", "OE1")],
    "GLH": [("CD", "OE1")],
    "HIE": [("CG", "CD2"), ("ND1", "CE1")],
    "HID": [("CG", "CD2"), ("CE1", "NE2")],
    "HIP": [("CG", "CD2"), ("ND1", "CE1")],
    "PHE": [("CG", "CD1"), ("CE1", "CZ"), ("CD2", "CE2")],
    "TYR": [("CG", "CD1"), ("CE1", "CZ"), ("CD2", "CE2")],
    "TRP": [("CG", "CD1"), ("CD2", "CE2"), ("CE3", "CZ3"), ("CZ2", "CH2")],
}
#: Formal charges of side chains in each residue.
FORMAL = {
    "ARG": {"NH2": 1},
    "ASP": {"OD2": -1},
    "GLU": {"OE2": -1},
    "HIP": {"ND1": 1},
    "LYS": {"NZ": 1},
    "CYM": {"SG": -1},
}
SS_DISTANCE = 2.5  # A
PEPTIDE_DISTANCE = 2.0  # A, C(i)-N(i+1) longer than this is a chain break
AMIDE_NH = 1.01  # A


class ProteinError(ValueError):
    """The structure cannot be prepared."""


@dataclass
class Report:
    residues: int = 0
    chains: int = 0
    hydrogens: int = 0
    added_heavy: list[str] = field(default_factory=list)
    histidines: dict[str, int] = field(default_factory=dict)
    disulfides: list[str] = field(default_factory=list)
    breaks: list[str] = field(default_factory=list)
    removed_water: int = 0
    removed_other: list[str] = field(default_factory=list)
    other_groups: list[str] = field(default_factory=list)
    duplicates: int = 0
    net_charge: int = 0

    def lines(self):
        out = [
            (
                f"{self.residues} residues in {self.chains} chain(s); {self.hydrogens} "
                f"hydrogens built; net charge {self.net_charge:+d}."
            )
        ]
        if self.histidines:
            out.append(
                "Histidines: "
                + ", ".join(f"{n} {k}" for k, n in sorted(self.histidines.items()))
                + "."
            )
        if self.disulfides:
            out.append(f"Disulfides: {', '.join(self.disulfides)}.")
        if self.breaks:
            out.append(
                f"Chain breaks (charged ends added): {', '.join(self.breaks)}. "
                "Check whether residues are missing there."
            )
        if self.added_heavy:
            shown = ", ".join(self.added_heavy[:12])
            more = (
                f" and {len(self.added_heavy) - 12} more"
                if len(self.added_heavy) > 12
                else ""
            )
            out.append(f"Missing heavy atoms added: {shown}{more}.")
        if self.duplicates:
            out.append(
                f"{self.duplicates} duplicate atoms (alternate locations) were dropped."
            )
        if self.removed_water:
            out.append(f"{self.removed_water} water molecules removed.")
        if self.removed_other:
            out.append(
                f"Removed: {', '.join(self.removed_other[:10])}"
                + (" ..." if len(self.removed_other) > 10 else "")
                + "."
            )
        if self.other_groups:
            out.append(
                f"Kept as they are (need their own hydrogens and parameters): "
                f"{', '.join(self.other_groups[:10])}"
                + (" ..." if len(self.other_groups) > 10 else "")
            )
        return out


@dataclass
class _Residue:
    key: tuple
    resname: str
    atoms: list[int]
    chain: str
    resnum: int


def _residues(system):
    groups = []
    for k in range(system.n_atoms):
        key = (system.chains[k], int(system.resnums[k]), system.resnames[k].upper())
        if not groups or groups[-1].key != key:
            groups.append(_Residue(key, key[2], [], key[0], key[1]))
        groups[-1].atoms.append(k)
    return groups


def _label(res):
    return f"{res.resname}{res.resnum}" + (f":{res.chain}" if res.chain.strip() else "")


def _fit(src, dst):
    """Rotation R and translation t with R @ src_i + t ~ dst_i (Kabsch)."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    h = (src - sc).T @ (dst - dc)
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, dc - r @ sc


class _Builder:
    """Places one residue's missing atoms from its template."""

    def __init__(self, tname):
        atoms, bonds, head, tail, positions, charge = TEMPLATES[tname]
        self.tname = tname
        self.names = [a[0] for a in atoms]
        self.elements = [a[1] for a in atoms]
        self.bonds = bonds
        self.head, self.tail = head, tail
        self.tpos = np.array(positions, dtype=float)
        self.charge = charge
        self.nbr = [[] for _ in atoms]
        for i, j in bonds:
            self.nbr[i].append(j)
            self.nbr[j].append(i)
        self.index = {n: k for k, n in enumerate(self.names)}

    def place(self, pos, prev_c=None):
        """Fill ``pos`` (template index -> xyz) for every atom; returns added indices."""
        added = []
        todo = [k for k in range(len(self.names)) if k not in pos]
        # heavy atoms first, so hydrogens see the finished heavy skeleton
        todo.sort(key=lambda k: self.elements[k] == "H")
        while todo:
            progress = False
            for k in list(todo):
                xyz = self._position(k, pos, prev_c)
                if xyz is None:
                    continue
                pos[k] = xyz
                added.append(k)
                todo.remove(k)
                progress = True
            if not progress:
                missing = ", ".join(self.names[k] for k in todo)
                raise ProteinError(
                    f"cannot place {missing}: too few atoms to start from"
                )
        return added

    def _position(self, k, pos, prev_c):
        parents = [p for p in self.nbr[k] if p in pos]
        if not parents:
            return None
        p = parents[0]
        if self.names[k] == "H" and p == self.head and prev_c is not None:
            ca = self.index.get("CA")
            if ca is not None and ca in pos:
                u = pos[p] - prev_c
                v = pos[p] - pos[ca]
                bis = u / np.linalg.norm(u) + v / np.linalg.norm(v)
                return pos[p] + AMIDE_NH * bis / np.linalg.norm(bis)
        frame = [p]
        for a in self.nbr[p]:
            if a != k and a in pos and len(frame) < 3:
                frame.append(a)
        if len(frame) < 3:
            for a in list(frame[1:]):
                for b in self.nbr[a]:
                    if b not in frame and b != k and b in pos and len(frame) < 3:
                        frame.append(b)
        if len(frame) < 3:
            return None
        src = self.tpos[frame]
        dst = np.array([pos[f] for f in frame])
        r, t = _fit(src, dst)
        return r @ self.tpos[k] + t


def _distance(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def prepare(system, histidine="HIE", remove_water=True, remove_other=False):
    """(new System, Report). Raises ProteinError when a residue cannot be rebuilt."""
    if histidine not in HISTIDINES:
        raise ValueError(f"histidine must be one of {HISTIDINES}")
    report = Report()
    groups = _residues(system)
    protein = [g for g in groups if g.resname in BASE]
    if not protein:
        raise ProteinError(
            "No amino-acid residues found. Open the protein from a PDB file (residue "
            "names are needed)."
        )
    xyz = system.xyz

    # heavy atoms by Amber name; hydrogens only remembered for the histidine state
    heavy = {}
    for g in protein:
        names = {}
        aliases = RESIDUE_ALIASES.get(BASE[g.resname], {})
        for k in g.atoms:
            if system.elements[k] == "H":
                continue
            name = system.atom_names[k].strip().upper()
            name = aliases.get(name, ALIASES.get(name, name))
            if name in names:
                report.duplicates += 1
                continue
            names[name] = k
        heavy[id(g)] = names

    def atom(g, name):
        k = heavy[id(g)].get(name)
        return None if k is None else xyz[k]

    # chain links and breaks
    linked_prev = {}
    for a, b in zip(protein, protein[1:]):  # noqa: RUF007 - MoleditPy supports Python 3.9
        c, n = atom(a, "C"), atom(b, "N")
        if a.chain == b.chain and c is not None and n is not None:
            if _distance(c, n) <= PEPTIDE_DISTANCE:
                linked_prev[id(b)] = a
                continue
            if BASE[a.resname] not in CAPS and BASE[b.resname] not in CAPS:
                report.breaks.append(f"{_label(a)}-{_label(b)}")
    starts = [g for g in protein if id(g) not in linked_prev]
    report.chains = len(starts)
    linked_next = {id(linked_prev[id(b)]): b for b in protein if id(b) in linked_prev}

    # disulfides
    cys = [
        g
        for g in protein
        if BASE[g.resname] in ("CYS", "CYM") and atom(g, "SG") is not None
    ]
    pairs = []
    for i, a in enumerate(cys):
        for b in cys[i + 1 :]:
            d = _distance(atom(a, "SG"), atom(b, "SG"))
            if d <= SS_DISTANCE:
                pairs.append((d, a, b))
    partner = {}
    for _d, a, b in sorted(pairs, key=lambda x: x[0]):
        if id(a) not in partner and id(b) not in partner:
            partner[id(a)], partner[id(b)] = b, a
            report.disulfides.append(f"{_label(a)}-{_label(b)}")

    def his_state(g):
        given = BASE[g.resname]
        if given != "HIS":
            return given
        hs = [k for k in g.atoms if system.elements[k] == "H"]
        found = set()
        for ring in ("ND1", "NE2"):
            n = atom(g, ring)
            if n is not None and any(_distance(xyz[h], n) < 1.3 for h in hs):
                found.add(ring)
        if found == {"ND1", "NE2"}:
            return "HIP"
        if found == {"ND1"}:
            return "HID"
        if found == {"NE2"}:
            return "HIE"
        return histidine

    # rebuild residue by residue
    out_el, out_xyz, out_names, out_res, out_num, out_chain, out_q = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    bonds = []
    new_index = {}  # old atom -> new atom (kept heavy atoms and other groups)
    placed = {}  # id(residue) -> (builder, {template index: new atom})
    for g in protein:
        base = BASE[g.resname]
        if base in ("CYS", "CYM"):
            base = "CYX" if id(g) in partner else ("CYM" if base == "CYM" else "CYS")
        elif base in ("HIS", "HID", "HIE", "HIP"):
            base = his_state(g)
            report.histidines[base] = report.histidines.get(base, 0) + 1
        first = id(g) not in linked_prev and base not in CAPS
        last = id(g) not in linked_next and base not in CAPS
        if first and last:
            raise ProteinError(
                f"{_label(g)} is a chain of one residue; it cannot be prepared"
            )
        tname = base
        if first and f"N{base}" in TEMPLATES:
            tname = f"N{base}"
        elif last and f"C{base}" in TEMPLATES:
            tname = f"C{base}"
        if tname not in TEMPLATES:
            raise ProteinError(f"{_label(g)}: no template {tname}")
        builder = _Builder(tname)
        pos = {}
        for name, k in heavy[id(g)].items():
            t = builder.index.get(name)
            if t is None:
                raise ProteinError(
                    f"{_label(g)}: atom {system.atom_names[k]} is not in residue {tname}"
                )
            pos[t] = xyz[k]
        prev = linked_prev.get(id(g))
        prev_c = atom(prev, "C") if prev is not None else None
        try:
            added = builder.place(pos, prev_c)
        except ProteinError as exc:
            raise ProteinError(f"{_label(g)}: {exc}") from exc
        for t in added:
            if builder.elements[t] == "H":
                report.hydrogens += 1
            else:
                report.added_heavy.append(f"{_label(g)} {builder.names[t]}")
        out_name = "CYX" if base == "CYX" else base
        formal = dict(FORMAL.get(base, {}))
        if first:
            formal["N"] = formal.get("N", 0) + 1
        if last and "OXT" in builder.index:
            formal["OXT"] = formal.get("OXT", 0) - 1
        if sum(formal.values()) != builder.charge:
            raise ProteinError(
                f"{_label(g)}: formal charges do not add up to {builder.charge}"
            )
        mapping = {}
        for t, name in enumerate(builder.names):
            mapping[t] = len(out_el)
            out_el.append(builder.elements[t])
            out_xyz.append(pos[t])
            out_names.append(name)
            out_res.append(out_name)
            out_num.append(g.resnum)
            out_chain.append(g.chain)
            out_q.append(formal.get(name, 0))
            k = heavy[id(g)].get(name)
            if k is not None:
                new_index[k] = mapping[t]
        doubles = {frozenset(p) for p in DOUBLE_BONDS.get(base, [])}
        if base not in ("NME", "NHE"):
            doubles.add(frozenset(("C", "O")))
        for i, j in builder.bonds:
            order = (
                2.0
                if frozenset((builder.names[i], builder.names[j])) in doubles
                else 1.0
            )
            bonds.append((mapping[i], mapping[j], order))
        placed[id(g)] = (builder, mapping)
        report.residues += 1
        report.net_charge += builder.charge

    for g in protein:
        prev = linked_prev.get(id(g))
        if prev is not None:
            b_prev, m_prev = placed[id(prev)]
            b_cur, m_cur = placed[id(g)]
            bonds.append((m_prev[b_prev.tail], m_cur[b_cur.head], 1.0))
    done = set()
    for g in protein:
        other = partner.get(id(g))
        if other is not None and id(other) not in done:
            b1, m1 = placed[id(g)]
            b2, m2 = placed[id(other)]
            bonds.append((m1[b1.index["SG"]], m2[b2.index["SG"]], 1.0))
            done.add(id(g))

    # everything else: waters (unless removed), ions, ligands, as they are
    others = [g for g in groups if g.resname not in BASE]
    for g in others:
        if remove_water and g.resname in WATER_NAMES:
            report.removed_water += 1
            continue
        if remove_other and g.resname not in WATER_NAMES:
            report.removed_other.append(_label(g))
            continue
        if g.resname not in WATER_NAMES:
            report.other_groups.append(_label(g))
        for k in g.atoms:
            new_index[k] = len(out_el)
            out_el.append(system.elements[k])
            out_xyz.append(xyz[k])
            out_names.append(system.atom_names[k])
            out_res.append(system.resnames[k])
            out_num.append(int(system.resnums[k]))
            out_chain.append(system.chains[k])
            out_q.append(int(system.charges[k]))
            report.net_charge += int(system.charges[k])
    protein_atoms = {k for g in protein for k in g.atoms}
    for i, j, order in system.bonds:
        if i in protein_atoms and j in protein_atoms:
            continue  # rebuilt from the templates
        if i in new_index and j in new_index:
            bonds.append((new_index[i], new_index[j], order))

    n = len(out_el)
    result = System(
        out_el,
        np.array(out_xyz, dtype=float).reshape(n, 3),
        bonds,
        np.array(out_q, dtype=int),
        perceive_molecules(n, bonds),
        out_res,
        np.array(out_num, dtype=int),
        out_names,
        out_chain,
        system.cell.copy() if system.cell is not None else None,
    )
    return result, report
