"""MD System Builder: box, packing, solvation and cell tools for MD systems in MoleditPy.

One dock panel with four tabs (Box, Pack, Solvate, Cell tools) sharing one packing
core, one cell module and one output path.  The result carries the suite-wide
``moleditpy_cell`` property that the LAMMPS / GROMACS / OpenMM input generators read.
"""

import json
import logging
import os

PLUGIN_NAME = "MD System Builder"
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DESCRIPTION = (
    "Build periodic MD systems: set the cell, pack molecules (Packmol-like), solvate "
    "with TIP3P water and ions, and wrap, unwrap, replicate or rescale the cell."
)
PLUGIN_CATEGORY = "Utility"
PLUGIN_TAGS = ["Utility", "Simulation"]
PLUGIN_DEPENDENCIES = []
PLUGIN_OPTIONAL_DEPENDENCIES = []
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=5.0.0, <6.0.0"
PLUGIN_LICENSE = "LGPL-3.0-or-later"

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")
WINDOW_ID = "md_system_builder_dock"

logger = logging.getLogger(__name__)

_context = None
_keeper = None
_panel_opened = False


def _load_settings():
    from .panel import default_settings

    settings = default_settings()
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as handle:
            saved = json.load(handle)
        if isinstance(saved, dict):
            settings.update({k: v for k, v in saved.items() if k in settings})
    except (OSError, ValueError):
        pass
    return settings


def _save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(settings, handle, indent=2)
    except OSError as exc:
        logger.debug("Could not save settings: %s", exc)


current_settings = {}


def _show_panel():
    """Open (or raise) the dock.  Not named ``run``: the host would add a second menu entry."""
    global _panel_opened
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDockWidget

    from .cif_viewer_link import find_cif_viewer_widget
    from .panel import BuilderPanel

    context = _context
    mw = context.get_main_window()
    _keeper.install()
    dock = context.get_window(WINDOW_ID)
    if dock is None:
        if not current_settings:
            current_settings.update(_load_settings())
        dock = QDockWidget(f"{PLUGIN_NAME} v{PLUGIN_VERSION}", mw)
        dock.setObjectName("MDSystemBuilderDock")
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        panel = BuilderPanel(
            context,
            _keeper,
            current_settings,
            get_cif_viewer=lambda: find_cif_viewer_widget(mw),
            parent=dock,
        )
        dock.setWidget(panel)
        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        context.register_window(WINDOW_ID, dock)
        _keeper.set_overlay(bool(current_settings.get("overlay", True)))
        app_quit = getattr(mw, "destroyed", None)
        if app_quit is not None:
            app_quit.connect(lambda *_: _shutdown(panel))
    _panel_opened = True
    dock.show()
    dock.raise_()


def _shutdown(panel):
    try:
        panel.shutdown()
        _save_settings(current_settings)
    except Exception:
        logger.debug("Ignored error", exc_info=True)


def initialize(context):
    global _context, _keeper
    from .host_link import CellKeeper

    _context = context
    _keeper = CellKeeper(context)
    _keeper.install()
    try:
        from PySide6.QtCore import QTimer

        # The 3D view may not exist yet while plugins load.
        QTimer.singleShot(0, _keeper.install)
    except Exception:
        logger.debug("Ignored error", exc_info=True)

    context.add_menu_action("Tools/MD System Builder...", _show_panel)

    def save_state():
        data = _keeper.state_for_save()
        if _panel_opened:
            dock = context.get_window(WINDOW_ID)
            panel = dock.widget() if dock is not None else None
            if panel is not None:
                current_settings.update(panel.read_settings())
                _save_settings(current_settings)
        return data

    def load_state(data):
        _keeper.remember_for_load(data)

    def handle_reset():
        _keeper.remember_for_load(None)

    context.register_save_handler(save_state)
    context.register_load_handler(load_state)
    context.register_document_reset_handler(handle_reset)
