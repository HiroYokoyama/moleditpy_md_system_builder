"""Metadata, entry point and a smoke test of the dock panel with real PySide6."""

import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INIT = os.path.join(ROOT, "md_system_builder", "__init__.py")


def metadata():
    with open(INIT, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("PLUGIN_"):
                values[name] = ast.literal_eval(node.value)
        if isinstance(node, ast.FunctionDef):
            values.setdefault("_functions", []).append(node.name)
    return values


def test_metadata():
    meta = metadata()
    assert meta["PLUGIN_NAME"] == "MD System Builder"
    assert meta["PLUGIN_VERSION"] == "0.1.1"
    assert meta["PLUGIN_AUTHOR"] == "HiroYokoyama"
    assert meta["PLUGIN_CATEGORY"] == "Utility"
    assert meta["PLUGIN_LICENSE"] == "LGPL-3.0-or-later"
    assert meta["PLUGIN_SUPPORTED_MOLEDITPY_VERSION"] == ">=5.0.0, <6.0.0"
    assert meta["PLUGIN_DEPENDENCIES"] == []


def test_no_run_function_and_an_initialize():
    functions = metadata()["_functions"]
    assert "initialize" in functions
    assert "run" not in functions


def test_package_name_matches_the_registry_id():
    name = metadata()["PLUGIN_NAME"].lower().replace(" ", "_")
    assert name == "md_system_builder"


class FakeContext:
    def __init__(self):
        self.menus = []
        self.handlers = {}
        self.windows = {}
        self.current_molecule = None
        self.plotter = None

    def add_menu_action(self, path, callback):
        self.menus.append(path)

    def register_save_handler(self, cb):
        self.handlers["save"] = cb

    def register_load_handler(self, cb):
        self.handlers["load"] = cb

    def register_document_reset_handler(self, cb):
        self.handlers["reset"] = cb

    def get_main_window(self):
        return None

    def show_status_message(self, *a):
        pass


def test_initialize_registers_menu_and_handlers():
    pytest.importorskip("PySide6")
    import md_system_builder as plugin

    ctx = FakeContext()
    plugin.initialize(ctx)
    assert ctx.menus == ["Tools/MD System Builder..."]
    assert set(ctx.handlers) == {"save", "load", "reset"}
    assert ctx.handlers["save"]() == {}
    ctx.handlers["load"]({"cell": "{}", "n_atoms": 3})
    ctx.handlers["reset"]()


def test_panel_builds_and_round_trips_settings(qapp):
    pytest.importorskip("rdkit")
    from md_system_builder.host_link import CellKeeper
    from md_system_builder.panel import BuilderPanel, default_settings

    ctx = FakeContext()
    panel = BuilderPanel(ctx, CellKeeper(ctx), default_settings())
    assert [panel.tabs.tabText(i) for i in range(panel.tabs.count())] == [
        "Protein",
        "Box",
        "Pack",
        "Solvate",
        "Cell tools",
    ]
    settings = panel.read_settings()
    settings["tolerance"] = 1.7
    settings["salt"] = "KCl"
    panel.apply_settings(settings)
    assert panel.read_settings()["tolerance"] == pytest.approx(1.7)
    assert panel.read_settings()["salt"] == "KCl"
    assert "No cell" in panel.cell_label.text()

    from md_system_builder import molio

    panel._add_component(molio.from_smiles("O", name="SOL"), "SOL", 5)
    assert panel.comp_table.rowCount() == 1
    panel.region_kind.setCurrentIndex(1)
    assert panel.components[0].region.kind == "inside_sphere"
    panel.solvent_mode.setCurrentIndex(0)
    assert not panel.solvent_smiles.isEnabled()
    panel.shutdown()
