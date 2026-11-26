"""Cell record, packing, solvation and cell tools (numpy only)."""

import json

import numpy as np
import pytest

from md_system_builder import celltools, packing, solvate
from md_system_builder.cellmeta import (
    PROP,
    PeriodicCell,
    bounding_box_cell,
    read_cell,
    write_cell,
)
from md_system_builder.system import System, concatenate, perceive_molecules


class FakeMol:
    def __init__(self):
        self.props = {}

    def HasProp(self, k):
        return k in self.props

    def GetProp(self, k):
        return self.props[k]

    def SetProp(self, k, v):
        self.props[k] = v

    def ClearProp(self, k):
        self.props.pop(k, None)


def water():
    return solvate.water_template()


def min_intermolecular(system):
    cell = system.cell
    best = np.inf
    mols = system.molecules()
    owner = np.empty(system.n_atoms, dtype=int)
    for m, atoms in enumerate(mols):
        owner[atoms] = m
    for i in range(system.n_atoms):
        d = cell.minimum_image(system.xyz - system.xyz[i])
        r = np.sqrt((d * d).sum(axis=1))
        r[owner == owner[i]] = np.inf
        best = min(best, r.min())
    return best


# -- cell record -------------------------------------------------------------


def test_cell_json_round_trip_matches_the_suite_format():
    cell = PeriodicCell.from_parameters([10, 11, 12], [80, 85, 95], origin=[1, 2, 3])
    data = json.loads(cell.to_json())
    assert set(data) == {"lattice", "origin", "pbc"}
    assert data["pbc"] == [True, True, True]
    back = PeriodicCell.from_json(cell.to_json())
    assert np.allclose(back.lattice, cell.lattice, atol=1e-7)
    assert np.allclose(back.origin, [1, 2, 3])
    (a, b, c), (al, be, ga) = back.parameters
    assert (round(a, 6), round(b, 6), round(c, 6)) == (10, 11, 12)
    assert (round(al, 4), round(be, 4), round(ga, 4)) == (80, 85, 95)


def test_read_and_write_cell_on_a_molecule():
    mol = FakeMol()
    assert read_cell(mol) is None
    write_cell(mol, PeriodicCell.from_lengths([5, 6, 7]))
    assert read_cell(mol).volume == pytest.approx(210)
    mol.SetProp(PROP, "not json")
    assert read_cell(mol) is None
    write_cell(mol, None)
    assert not mol.HasProp(PROP)


def test_invalid_cells_are_rejected():
    with pytest.raises(ValueError):
        PeriodicCell(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        PeriodicCell.from_dict({"lattice": [[1, 0], [0, 1]]})


def test_bounding_box_padding_and_cubic():
    xyz = np.array([[0, 0, 0], [4, 2, 1]], dtype=float)
    cell = bounding_box_cell(xyz, 3.0)
    assert np.allclose(np.diag(cell.lattice), [10, 8, 7])
    assert np.allclose(cell.origin, [-3, -3, -3])
    cube = bounding_box_cell(xyz, 3.0, cubic=True)
    assert np.allclose(np.diag(cube.lattice), [10, 10, 10])
    assert np.allclose(cube.centre, (xyz.min(0) + xyz.max(0)) / 2)


def test_minimum_image_triclinic():
    cell = PeriodicCell.from_parameters([10, 10, 10], [90, 90, 120])
    delta = cell.lattice[0] * 0.9
    assert np.linalg.norm(cell.minimum_image(delta.reshape(1, 3))) == pytest.approx(1.0)


# -- system -------------------------------------------------------------------


def test_perceive_molecules_orders_by_first_atom():
    ids = perceive_molecules(5, [(3, 4, 1.0), (0, 2, 1.0)])
    assert ids.tolist() == [0, 1, 0, 2, 2]


def test_concatenate_and_subset_keep_bonds_and_residues():
    a = water()
    b = water()
    b.xyz = b.xyz + 5
    joined = concatenate([a, b])
    assert joined.n_atoms == 6 and joined.n_molecules == 2
    assert (3, 4, 1.0) in joined.bonds
    assert joined.resnums.tolist() == [1, 1, 1, 2, 2, 2]
    sub = joined.subset([3, 4, 5])
    assert sub.bonds == [(0, 1, 1.0), (0, 2, 1.0)]
    assert sub.mol_index.tolist() == [0, 0, 0]


# -- packing ------------------------------------------------------------------


def test_pack_respects_tolerance_with_periodic_images():
    cell = PeriodicCell.from_lengths([20, 20, 20])
    result = packing.pack(
        cell, [packing.Component(water(), 120, "SOL")], tolerance=2.0, seed=1
    )
    assert result.placed == [120]
    assert min_intermolecular(result.system) >= 2.0 - 1e-9
    assert result.system.residue_counts() == [("SOL", 120)]


def test_pack_is_reproducible_with_a_seed():
    cell = PeriodicCell.from_lengths([15, 15, 15])
    comp = [packing.Component(water(), 30, "SOL")]
    one = packing.pack(cell, comp, seed=7).system.xyz
    two = packing.pack(cell, comp, seed=7).system.xyz
    three = packing.pack(cell, comp, seed=8).system.xyz
    assert np.allclose(one, two)
    assert not np.allclose(one, three)


def test_pack_too_full_raises_a_clear_error():
    cell = PeriodicCell.from_lengths([8, 8, 8])
    with pytest.raises(packing.PackingError, match="too full"):
        packing.pack(
            cell, [packing.Component(water(), 500, "SOL")], seed=1, max_attempts=30
        )


def test_pack_triclinic_cell():
    cell = PeriodicCell.from_parameters([18, 18, 18], [70, 80, 110])
    result = packing.pack(cell, [packing.Component(water(), 60, "SOL")], seed=2)
    assert min_intermolecular(result.system) >= 2.0 - 1e-9


def test_fixed_solute_is_centred_and_avoided():
    cell = PeriodicCell.from_lengths([20, 20, 20])
    solute = water()
    result = packing.pack(
        cell, [packing.Component(water(), 50, "SOL")], fixed=solute, seed=3
    )
    sys_ = result.system
    assert np.allclose(sys_.xyz[:3].mean(axis=0), cell.centre)
    assert sys_.n_molecules == 51
    assert min_intermolecular(sys_) >= 2.0 - 1e-9


@pytest.mark.parametrize(
    "region, check",
    [
        (
            packing.Region(packing.REGION_INSIDE_SPHERE, radius=6.0),
            lambda r: r <= 6.0 + 1e-9,
        ),
        (
            packing.Region(packing.REGION_OUTSIDE_SPHERE, radius=6.0),
            lambda r: r >= 6.0 - 1e-9,
        ),
    ],
)
def test_sphere_regions(region, check):
    cell = PeriodicCell.from_lengths([24, 24, 24])
    result = packing.pack(cell, [packing.Component(water(), 20, "SOL", region)], seed=4)
    r = np.linalg.norm(result.system.xyz - cell.centre, axis=1)
    assert np.all(check(r))


def test_slab_regions():
    cell = PeriodicCell.from_lengths([20, 20, 30])
    inside = packing.Region(packing.REGION_INSIDE_SLAB, axis=2, low=5, high=12)
    outside = packing.Region(packing.REGION_OUTSIDE_SLAB, axis=2, low=5, high=25)
    result = packing.pack(
        cell,
        [
            packing.Component(water(), 15, "A", inside),
            packing.Component(water(), 15, "B", outside),
        ],
        seed=5,
    )
    z = result.system.xyz[:, 2]
    assert np.all((z[:45] >= 5) & (z[:45] <= 12))
    assert not np.any((z[45:] >= 5) & (z[45:] <= 25))


def test_region_validation():
    with pytest.raises(ValueError):
        packing.Region("cube")
    with pytest.raises(ValueError):
        packing.Region(packing.REGION_INSIDE_SLAB, low=3, high=1)


def test_density_counts():
    # 1 g/cm3 water in 1000 A^3 is 33.4 molecules
    assert packing.count_for_density(1.0, 1000.0, 18.015) == 33
    comps = [packing.Component(water(), 1, "A"), packing.Component(water(), 1, "B")]
    assert packing.mixture_counts(1.0, 3000.0, comps, [1, 1]) == [50, 50]


def test_cancel_stops_packing():
    cell = PeriodicCell.from_lengths([30, 30, 30])
    with pytest.raises(packing.Cancelled):
        packing.pack(
            cell, [packing.Component(water(), 100, "SOL")], cancelled=lambda: True
        )


# -- solvation ----------------------------------------------------------------


def test_tiled_water_density_and_no_clashes():
    cell = PeriodicCell.from_lengths([40, 40, 40])
    waters = solvate.tile_water(cell)
    density = waters.total_mass() / cell.volume / packing.AVOGADRO_A3
    assert 0.95 < density < 1.02
    oxygens = waters.xyz[::3]
    frac = cell.to_fractional(oxygens)
    assert np.all((frac >= 0) & (frac < 1))
    assert waters.residue_counts() == [("SOL", waters.n_molecules)]
    assert len(waters.bonds) == 2 * waters.n_molecules


def test_tiled_water_removes_waters_near_the_solute():
    cell = PeriodicCell.from_lengths([25, 25, 25])
    solute = System(["C"], np.array([cell.centre]))
    waters = solvate.tile_water(cell, solute, solute_cutoff=3.0)
    d = np.linalg.norm(cell.minimum_image(waters.xyz - cell.centre), axis=1)
    assert d.min() >= 3.0


def test_tiled_water_in_a_triclinic_cell():
    cell = PeriodicCell.from_parameters([30, 30, 30], [90, 90, 120])
    waters = solvate.tile_water(cell)
    density = waters.total_mass() / cell.volume / packing.AVOGADRO_A3
    assert density > 0.9


@pytest.mark.parametrize(
    "charge, conc, n_water, expected",
    [
        (0, 0.0, 1000, (0, 0)),
        (-3, 0.0, 1000, (3, 0)),
        (2, 0.0, 1000, (0, 2)),
        (0, 0.15, 1107, (3, 3)),
        (-1, 0.15, 1107, (4, 3)),
    ],
)
def test_plan_ions(charge, conc, n_water, expected):
    plan = solvate.plan_ions(charge, n_water, "NaCl", True, conc)
    assert (plan.n_cation, plan.n_anion) == expected


def test_place_ions_neutralises_and_keeps_distance():
    cell = PeriodicCell.from_lengths([30, 30, 30])
    solute = System(["C"], np.array([cell.centre]), charges=np.array([-2]))
    waters = solvate.tile_water(cell, solute)
    plan = solvate.plan_ions(-2, waters.n_molecules, "KCl", True, 0.15)
    out = solvate.place_ions(waters, plan, solute, cell, 5.0, 5.0, seed=1)
    assert out.net_charge() == 2
    assert out.n_molecules == waters.n_molecules
    names = [name for name, _ in out.residue_counts()]
    assert names == ["SOL", "K", "CL"]
    ions = out.xyz[np.array([r in ("K", "CL") for r in out.resnames])]
    d = np.linalg.norm(cell.minimum_image(ions - cell.centre), axis=1)
    assert d.min() >= 5.0


def test_place_ions_relaxes_distances_in_a_small_box():
    cell = PeriodicCell.from_lengths([15, 15, 15])
    waters = solvate.tile_water(cell)
    plan = solvate.IonPlan("Na+", "Cl-", 10, 10)
    out = solvate.place_ions(waters, plan, None, cell, 5.0, 8.0, seed=1)
    assert out.net_charge() == 0
    with pytest.raises(ValueError):
        solvate.place_ions(waters, solvate.IonPlan("Na+", "Cl-", 999, 999), None, cell)


# -- cell tools ---------------------------------------------------------------


def packed(seed=1, cell=None):
    cell = cell or PeriodicCell.from_lengths([15, 16, 17])
    return packing.pack(cell, [packing.Component(water(), 25, "SOL")], seed=seed).system


def bond_lengths(system):
    return np.array(
        [np.linalg.norm(system.xyz[i] - system.xyz[j]) for i, j, _ in system.bonds]
    )


def test_wrap_and_unwrap():
    sys_ = packed()
    sys_.xyz = sys_.xyz + np.array([7.0, -3.0, 11.0])
    atoms = celltools.wrap_atoms(sys_)
    frac = sys_.cell.to_fractional(atoms.xyz)
    assert np.all((frac >= -1e-9) & (frac < 1 + 1e-9))
    whole = celltools.unwrap(atoms)
    assert bond_lengths(whole).max() < 1.0
    mols = celltools.wrap_molecules(sys_)
    assert bond_lengths(mols).max() < 1.0
    for idx in mols.molecules():
        c = mols.cell.to_fractional(mols.xyz[idx].mean(axis=0))
        assert np.all((c >= 0) & (c < 1))


def test_centre():
    sys_ = packed()
    out = celltools.centre(sys_, atoms=[0, 1, 2])
    assert np.allclose(out.xyz[:3].mean(axis=0), sys_.cell.centre)


def test_supercell_counts_and_order():
    sys_ = packed()
    sc = celltools.supercell(sys_, [2, 2, 1])
    assert sc.n_atoms == 4 * sys_.n_atoms
    assert sc.cell.volume == pytest.approx(4 * sys_.cell.volume)
    assert sc.residue_counts() == [("SOL", 100)]
    with pytest.raises(ValueError):
        celltools.supercell(sys_, [0, 1, 1])


def test_scale_modes():
    sys_ = packed()
    rigid = celltools.scale(sys_, [1.2, 1.2, 1.2], "molecules")
    assert rigid.cell.volume == pytest.approx(sys_.cell.volume * 1.728)
    assert np.allclose(bond_lengths(rigid), bond_lengths(sys_))
    atoms = celltools.scale(sys_, [2, 1, 1], "atoms")
    assert np.allclose(
        atoms.xyz[:, 0],
        sys_.cell.origin[0] + 2 * (sys_.xyz[:, 0] - sys_.cell.origin[0]),
    )
    none = celltools.scale(sys_, [2, 1, 1], "none")
    assert np.allclose(none.xyz, sys_.xyz)
    with pytest.raises(ValueError):
        celltools.scale(sys_, [0, 1, 1])


def test_orthorhombic_from_hexagonal():
    cell = PeriodicCell.from_parameters([16, 16, 16], [90, 90, 120])
    sys_ = packed(cell=cell)
    out = celltools.to_orthorhombic(sys_)
    assert out.cell.is_orthorhombic()
    assert out.n_atoms == 2 * sys_.n_atoms
    assert out.cell.volume == pytest.approx(2 * cell.volume)
    assert bond_lengths(out).max() < 1.0
    assert min_intermolecular(out) > 1.9


def test_orthorhombic_impossible_for_general_triclinic():
    cell = PeriodicCell.from_parameters([10, 11, 12], [80, 85, 95])
    sys_ = System(["Ar"], np.array([[1.0, 1.0, 1.0]]), cell=cell)
    with pytest.raises(ValueError, match="No orthorhombic cell"):
        celltools.to_orthorhombic(sys_)


def test_tools_need_a_cell():
    with pytest.raises(ValueError, match="no periodic cell"):
        celltools.wrap_atoms(water())
