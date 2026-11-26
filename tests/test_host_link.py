"""CellKeeper: the cell survives the host's undo snapshots and project reloads."""

import copy

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from md_system_builder import host_link
from md_system_builder.cellmeta import PROP, PeriodicCell, write_cell


class FakeHost:
    """The parts of MainWindow / PluginContext the keeper uses, snapshotting like the host."""

    def __init__(self):
        self.mol = None
        manager = type("M", (), {})()
        manager.undo_stack = []
        manager.redo_stack = []
        manager.update_undo_redo_actions = lambda: None
        self.edit_actions_manager = manager
        self.plugin_manager = type("P", (), {"custom_3d_styles": {}})()
        self.modified = 0

    # PluginContext
    def get_main_window(self):
        return self

    @property
    def current_molecule(self):
        return self.mol

    @current_molecule.setter
    def current_molecule(self, mol):
        self.mol = mol

    plotter = None

    def mark_project_modified(self):
        self.modified += 1

    def _snapshot(self):
        return {
            "mol_3d": self.mol.ToBinary() if self.mol is not None else None,
            "mol_3d_atom_ids": [None] * self.mol.GetNumAtoms()
            if self.mol is not None
            else None,
        }

    def push_undo_checkpoint(self):
        stack = self.edit_actions_manager.undo_stack
        state = self._snapshot()
        if stack and stack[-1]["mol_3d"] == state["mol_3d"]:
            return  # the host skips identical states
        stack.append(copy.deepcopy(state))
        self.edit_actions_manager.redo_stack.clear()

    def _restore(self, state):
        self.mol = Chem.Mol(state["mol_3d"]) if state["mol_3d"] else None

    def undo(self):
        stack = self.edit_actions_manager.undo_stack
        self.edit_actions_manager.redo_stack.append(stack.pop())
        self._restore(stack[-1])

    def redo(self):
        state = self.edit_actions_manager.redo_stack.pop()
        self.edit_actions_manager.undo_stack.append(state)
        self._restore(state)


def mol(smiles="CCO"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    from rdkit.Chem import AllChem

    AllChem.EmbedMolecule(m, randomSeed=1)
    return m


@pytest.fixture
def host():
    h = FakeHost()
    h.mol = mol()
    h.push_undo_checkpoint()
    return h


def set_cell(host, keeper, lengths):
    new = Chem.Mol(host.mol)
    write_cell(new, PeriodicCell.from_lengths(lengths))
    host.mol = new
    keeper.checkpoint()


def test_cell_only_edits_become_undo_steps(host):
    keeper = host_link.CellKeeper(host)
    set_cell(host, keeper, [10, 10, 10])
    assert len(host.edit_actions_manager.undo_stack) == 2  # added by the keeper
    set_cell(host, keeper, [20, 20, 20])
    assert len(host.edit_actions_manager.undo_stack) == 3

    host.undo()
    assert not host.mol.HasProp(PROP)
    keeper.sync()
    assert keeper.current_cell().volume == pytest.approx(1000)
    host.undo()
    keeper.sync()
    assert not host.mol.HasProp(PROP)
    host.redo()
    keeper.sync()
    host.redo()
    keeper.sync()
    assert keeper.current_cell().volume == pytest.approx(8000)


def test_new_molecule_steps_are_tagged(host):
    keeper = host_link.CellKeeper(host)
    big = mol("CCCCCC")
    write_cell(big, PeriodicCell.from_lengths([30, 30, 30]))
    host.mol = big
    keeper.checkpoint()
    assert len(host.edit_actions_manager.undo_stack) == 2
    host.undo()
    keeper.sync()
    assert not host.mol.HasProp(PROP)
    host.redo()
    keeper.sync()
    assert keeper.current_cell().volume == pytest.approx(27000)


def test_host_pushed_states_get_the_current_cell(host):
    keeper = host_link.CellKeeper(host)
    set_cell(host, keeper, [12, 12, 12])
    # A later host edit (e.g. an atom drag) pushes an untagged state for the same molecule.
    moved = Chem.Mol(host.mol)
    moved.GetConformer().SetAtomPosition(0, (5.0, 5.0, 5.0))
    host.mol = moved
    host.push_undo_checkpoint()
    keeper.sync()
    assert host_link.STATE_KEY in host.edit_actions_manager.undo_stack[-1]
    host.undo()
    host.redo()
    keeper.sync()
    assert keeper.current_cell().volume == pytest.approx(1728)


def test_project_round_trip(host):
    keeper = host_link.CellKeeper(host)
    set_cell(host, keeper, [11, 12, 13])
    saved = keeper.state_for_save()
    assert saved["n_atoms"] == host.mol.GetNumAtoms()

    fresh = FakeHost()
    other = host_link.CellKeeper(fresh)
    other.remember_for_load(
        saved
    )  # the host calls load handlers before restoring the molecule
    fresh.mol = Chem.Mol(host.mol.ToBinary())
    assert not fresh.mol.HasProp(PROP)
    other.sync()
    assert other.current_cell().volume == pytest.approx(11 * 12 * 13)


def test_project_cell_is_not_put_on_a_different_molecule(host):
    keeper = host_link.CellKeeper(host)
    keeper.remember_for_load(
        {"cell": PeriodicCell.from_lengths([5, 5, 5]).to_json(), "n_atoms": 999}
    )
    host.mol = mol("C")
    keeper.sync()
    assert not host.mol.HasProp(PROP)


def test_save_without_a_cell_is_empty(host):
    assert host_link.CellKeeper(host).state_for_save() == {}


def test_large_system_viewer_detection():
    h = FakeHost()
    assert not host_link.large_system_viewer_installed(h)
    h.plugin_manager.custom_3d_styles[host_link.LSV_STYLE] = {}
    assert host_link.large_system_viewer_installed(h)


def test_open_with_host_uses_the_command_line_loader():
    opened = []
    mw = type("W", (), {})()
    mw.init_manager = type(
        "I", (), {"load_command_line_file": staticmethod(opened.append)}
    )()
    assert host_link.open_with_host(mw, "x.pdb")
    assert opened == ["x.pdb"]
    assert not host_link.open_with_host(object(), "x.pdb")
