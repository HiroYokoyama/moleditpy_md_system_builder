"""RDKit round trip, component input and PDB/GRO/XYZ output."""

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from md_system_builder import molio, packing, solvate
from md_system_builder.cellmeta import PROP, PeriodicCell
from md_system_builder.system import from_rdkit, to_rdkit


def small_system():
    cell = PeriodicCell.from_parameters([20, 21, 22], [90, 90, 110])
    meoh = molio.from_smiles("CO", name="MOH")
    result = packing.pack(
        cell,
        [
            packing.Component(meoh, 3, "MOH"),
            packing.Component(solvate.water_template(), 4, "SOL"),
        ],
        seed=1,
    )
    return result.system


def test_smiles_component_has_hydrogens_and_bonds():
    tpl = molio.from_smiles("CCO", name="ETH")
    assert tpl.n_atoms == 9
    assert len(tpl.bonds) == 8
    assert set(tpl.resnames) == {"ETH"}
    with pytest.raises(ValueError):
        molio.from_smiles("C1CC")


def test_round_trip_keeps_cell_residues_and_charges():
    sys_ = small_system()
    sys_.charges[0] = 0
    mol = to_rdkit(sys_)
    assert mol.HasProp(PROP)
    info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
    assert info.GetResidueName().strip() == "MOH"
    back = from_rdkit(mol)
    assert back.residue_counts() == [("MOH", 3), ("SOL", 4)]
    assert np.allclose(back.xyz, sys_.xyz)
    assert np.allclose(back.cell.lattice, sys_.cell.lattice, atol=1e-6)
    assert back.n_molecules == 7


def test_residue_info_survives_the_hosts_binary_snapshot():
    """The host snapshots with ToBinary(): the cell is dropped, residues are kept."""
    mol = to_rdkit(small_system())
    restored = Chem.Mol(mol.ToBinary())
    assert not restored.HasProp(PROP)
    assert from_rdkit(restored).residue_counts() == [("MOH", 3), ("SOL", 4)]


def test_ions_get_formal_charges():
    cell = PeriodicCell.from_lengths([20, 20, 20])
    waters = solvate.tile_water(cell)
    out = solvate.place_ions(
        waters, solvate.IonPlan("Na+", "Cl-", 1, 2), None, cell, seed=1
    )
    mol = to_rdkit(out)
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == -1


def test_from_file_formats(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CC"))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(mol, randomSeed=1)
    sdf = tmp_path / "ethane.sdf"
    Chem.MolToMolFile(mol, str(sdf))
    xyz = tmp_path / "ethane.xyz"
    Chem.MolToXYZFile(mol, str(xyz))
    pdb = tmp_path / "ethane.pdb"
    Chem.MolToPDBFile(mol, str(pdb))
    for path in (sdf, xyz, pdb):
        tpl = molio.from_file(str(path))
        assert tpl.n_atoms == 8
        assert len(tpl.bonds) == 7, path.suffix
    with pytest.raises(ValueError):
        molio.from_file(str(tmp_path / "missing.sdf"))


def test_pdb_has_cryst1_and_matches_standard_orientation(tmp_path):
    sys_ = small_system()
    path = tmp_path / "s.pdb"
    molio.write_system(sys_, str(path))
    text = path.read_text()
    assert text.splitlines()[1].startswith(
        "CRYST1   20.000   21.000   22.000  90.00  90.00 110.00"
    )
    back = Chem.MolFromPDBFile(str(path), removeHs=False, sanitize=False)
    assert back.GetNumAtoms() == sys_.n_atoms
    # Distances are invariant under the re-orientation.
    pos = back.GetConformer().GetPositions()
    assert np.linalg.norm(pos[0] - pos[1]) == pytest.approx(
        np.linalg.norm(sys_.xyz[0] - sys_.xyz[1]), abs=2e-3
    )


def test_gro_triclinic_box_line(tmp_path):
    sys_ = small_system()
    text = molio.gro_text(sys_)
    lines = text.splitlines()
    assert int(lines[1]) == sys_.n_atoms
    box = [float(v) for v in lines[-1].split()]
    assert len(box) == 9
    assert box[0] == pytest.approx(2.0)
    assert box[3] == box[4] == box[6] == 0.0  # v1(y), v1(z), v2(z)
    assert lines[2][5:10].strip() == "MOH"


def test_gro_orthorhombic_box_line():
    sys_ = solvate.tile_water(PeriodicCell.from_lengths([20, 20, 20]))
    assert molio.gro_text(sys_).splitlines()[-1].split() == [
        "2.00000",
        "2.00000",
        "2.00000",
    ]


def test_xyz_extended_comment(tmp_path):
    sys_ = small_system()
    text = molio.xyz_text(sys_)
    assert 'Lattice="20.000000 0.000000 0.000000' in text.splitlines()[1]
    assert 'pbc="T T T"' in text
    with pytest.raises(ValueError):
        molio.write_system(sys_, str(tmp_path / "x.cif"))
