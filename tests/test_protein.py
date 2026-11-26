"""Protein preparation: hydrogens at pH 7, chain ends, disulfides, missing atoms."""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem
from rdkit.Chem import AllChem

from md_system_builder import protein
from md_system_builder.protein_templates import TEMPLATES
from md_system_builder.system import System, from_rdkit, to_rdkit

SEQUENCE = "ACDEHKRC"  # Cys2 and Cys8 joined by a disulfide


@pytest.fixture(scope="module")
def peptide():
    """Heavy atoms of ACDEHKRC with an S-S bond, as a PDB file would give them."""
    mol = Chem.RWMol(Chem.MolFromSequence(SEQUENCE))
    sg = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "S"]
    mol.AddBond(sg[0], sg[1], Chem.BondType.SINGLE)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    return from_rdkit(Chem.RemoveHs(mol))


def _subset(system, keep):
    keep = [k for k in range(system.n_atoms) if keep(k)]
    return system.subset(np.array(keep))


def _atom(system, resnum, name):
    for k in range(system.n_atoms):
        if int(system.resnums[k]) == resnum and system.atom_names[k] == name:
            return k
    raise KeyError((resnum, name))


def test_hydrogens_charges_and_disulfide(peptide):
    out, report = protein.prepare(peptide)
    names = [out.resnames[_atom(out, r, "CA")] for r in range(1, 9)]
    assert names == ["ALA", "CYX", "ASP", "GLU", "HIE", "LYS", "ARG", "CYX"]
    assert report.disulfides == ["CYS2:A-CYS8:A"]
    assert report.histidines == {"HIE": 1}
    assert report.net_charge == 0 and int(out.charges.sum()) == 0
    expected_h = sum(
        e == "H"
        for tname in ("NALA", "CYX", "ASP", "GLU", "HIE", "LYS", "ARG", "CCYX")
        for _n, e in TEMPLATES[tname][0]
    )
    assert report.hydrogens == expected_h
    assert sum(e == "H" for e in out.elements) == expected_h
    # charged N terminus and C terminus, charged side chains
    assert out.charges[_atom(out, 1, "N")] == 1
    assert out.charges[_atom(out, 8, "OXT")] == -1
    assert out.charges[_atom(out, 6, "NZ")] == 1
    assert out.charges[_atom(out, 3, "OD2")] == -1
    # the S-S bond and every hydrogen bonded once, at a sensible length
    s1, s2 = _atom(out, 2, "SG"), _atom(out, 8, "SG")
    pairs = {frozenset((i, j)) for i, j, _o in out.bonds}
    assert frozenset((s1, s2)) in pairs
    for k, el in enumerate(out.elements):
        if el != "H":
            continue
        partners = [j if i == k else i for i, j, _o in out.bonds if k in (i, j)]
        assert len(partners) == 1
        length = np.linalg.norm(out.xyz[k] - out.xyz[partners[0]])
        assert 0.95 < length < 1.12
    assert out.n_molecules == 1


def test_amide_hydrogen_is_in_the_peptide_plane(peptide):
    out, _ = protein.prepare(peptide)
    c, n, ca, h = (
        out.xyz[_atom(out, 3, a)] if a != "C" else out.xyz[_atom(out, 2, "C")]
        for a in ("C", "N", "CA", "H")
    )
    normal = np.cross(c - n, ca - n)
    assert abs(np.dot(h - n, normal / np.linalg.norm(normal))) < 0.05
    assert np.linalg.norm(h - n) == pytest.approx(protein.AMIDE_NH, abs=1e-6)


def test_rdkit_round_trip_keeps_names_and_charge(peptide):
    out, _ = protein.prepare(peptide)
    mol = to_rdkit(out)
    assert Chem.GetFormalCharge(mol) == 0
    assert mol.GetNumAtoms() == out.n_atoms
    info = mol.GetAtomWithIdx(_atom(out, 5, "HE2")).GetPDBResidueInfo()
    assert info.GetResidueName() == "HIE"
    again, report = protein.prepare(from_rdkit(mol))
    assert again.n_atoms == out.n_atoms
    assert report.histidines == {"HIE": 1}  # read from the existing ring hydrogen


@pytest.mark.parametrize(("choice", "charge"), [("HID", 0), ("HIP", 1)])
def test_histidine_choice(peptide, choice, charge):
    out, report = protein.prepare(peptide, histidine=choice)
    assert report.histidines == {choice: 1}
    assert report.net_charge == charge == int(out.charges.sum())


def test_missing_side_chain_atom_and_oxt_are_added(peptide):
    lys_nz = _atom(peptide, 6, "NZ")
    oxt = _atom(peptide, 8, "OXT")
    cut = _subset(peptide, lambda k: k not in (lys_nz, oxt))
    out, report = protein.prepare(cut)
    assert report.added_heavy == ["LYS6:A NZ", "CYS8:A OXT"]
    nz, ce = _atom(out, 6, "NZ"), _atom(out, 6, "CE")
    assert np.linalg.norm(out.xyz[nz] - out.xyz[ce]) == pytest.approx(1.47, abs=0.05)
    assert report.net_charge == 0


def test_chain_break_gives_two_charged_chains(peptide):
    cut = _subset(peptide, lambda k: int(peptide.resnums[k]) != 4)
    out, report = protein.prepare(cut)
    assert report.chains == 2
    assert report.breaks == ["ASP3:A-HIS5:A"]
    assert out.charges[_atom(out, 5, "N")] == 1
    assert out.charges[_atom(out, 3, "OXT")] == -1
    assert out.n_molecules == 1  # still joined by the disulfide


def test_free_cysteine_keeps_its_hydrogen(peptide):
    far = peptide.copy()
    for k in range(far.n_atoms):
        if int(far.resnums[k]) >= 7:
            far.xyz[k] += [0.0, 0.0, 30.0]
    out, report = protein.prepare(far)
    assert report.disulfides == []
    assert report.breaks == ["LYS6:A-ARG7:A"]
    assert out.resnames[_atom(out, 2, "HG")] == "CYS"


def test_waters_and_other_groups(peptide):
    water = System(["O", "H", "H"], [[40, 0, 0], [40.96, 0, 0], [39.76, 0.93, 0]],
                   [(0, 1, 1.0), (0, 2, 1.0)], resnames=["HOH"] * 3,
                   resnums=[101] * 3, atom_names=["O", "H1", "H2"])  # fmt: skip
    ion = System(["Na"], [[0, 40, 0]], charges=[1], resnames=["NA"], resnums=[102],
                 atom_names=["NA"])  # fmt: skip
    from md_system_builder.system import concatenate

    full = concatenate([peptide, water, ion])
    out, report = protein.prepare(full)
    assert report.removed_water == 1
    assert [g[:2] for g in report.other_groups] == ["NA"]
    assert "NA" in out.resnames and "HOH" not in out.resnames
    assert report.net_charge == 1
    out, report = protein.prepare(full, remove_water=False, remove_other=True)
    assert "HOH" in out.resnames and "NA" not in out.resnames
    assert [g[:2] for g in report.removed_other] == ["NA"]


def test_duplicate_alternate_locations_are_dropped(peptide):
    ca = _atom(peptide, 4, "CA")
    order = list(range(peptide.n_atoms))
    order.insert(ca + 1, ca)  # a second CA right after the first, as altloc B
    doubled = System(
        [peptide.elements[k] for k in order],
        peptide.xyz[order] + 0.01,
        [],
        resnames=[peptide.resnames[k] for k in order],
        resnums=peptide.resnums[order],
        atom_names=[peptide.atom_names[k] for k in order],
        chains=[peptide.chains[k] for k in order],
    )
    out, report = protein.prepare(doubled)
    assert report.duplicates == 1
    assert out.n_atoms == protein.prepare(peptide)[0].n_atoms


def test_single_residue_chain_is_an_error(peptide):
    far = peptide.copy()
    for k in range(far.n_atoms):
        if int(far.resnums[k]) == 8:
            far.xyz[k] += [0.0, 0.0, 30.0]
    with pytest.raises(protein.ProteinError, match="chain of one residue"):
        protein.prepare(far)


def test_errors(peptide):
    with pytest.raises(protein.ProteinError, match="No amino-acid"):
        protein.prepare(System(["O"], [[0, 0, 0]], resnames=["HOH"]))
    odd = peptide.copy()
    odd.atom_names[_atom(odd, 3, "CB")] = "XX"
    with pytest.raises(protein.ProteinError, match="ASP3:A: atom XX"):
        protein.prepare(odd)
    with pytest.raises(ValueError):
        protein.prepare(peptide, histidine="HSX")


def test_report_lines(peptide):
    _out, report = protein.prepare(peptide)
    text = " ".join(report.lines())
    assert "8 residues in 1 chain(s)" in text and "Disulfides: CYS2:A-CYS8:A" in text
