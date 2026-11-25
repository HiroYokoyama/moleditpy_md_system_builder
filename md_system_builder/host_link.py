"""Everything that touches MoleditPy beyond the public PluginContext calls.

Why this exists: the host snapshots ``current_mol`` with ``Mol.ToBinary()`` for undo
and project files, and RDKit's default pickle options drop molecule properties, so
``moleditpy_cell`` is lost on every undo, redo and project reload (PDB residue info
survives).  The :class:`CellKeeper` therefore

* tags each undo state with the cell of the molecule it holds (an extra key in the
  host's state dict, which the host ignores when restoring),
* puts the cell back on the molecule after undo/redo, and after a project load from
  the record kept by the plugin's save handler,
* redraws the cell box whenever the host has redrawn the scene and removed it.

It watches the host's render window (a VTK observer) instead of replacing host
methods, so it coexists with other plugins that wrap ``draw_molecule_3d``.
"""

from __future__ import annotations

import copy
import logging

import numpy as np

from .cellmeta import PROP, PeriodicCell, read_cell

logger = logging.getLogger(__name__)

STATE_KEY = "moleditpy_md_system_builder_cell"
ACTOR_PREFIX = "md_system_builder_cell"
LSV_STYLE = "Large System (Fast)"
LSV_HOST_LINK_LIMIT = 50000  # Large System Viewer's default view-only threshold (atoms)
LSV_SWITCH_THRESHOLD = 2000  # where the Large System Viewer offers its fast style


def _mw(context):
    try:
        return context.get_main_window()
    except Exception:  # noqa: BLE001 - host API guard
        return None


def _stacks(mw):
    manager = getattr(mw, "edit_actions_manager", None)
    undo = getattr(manager, "undo_stack", None)
    redo = getattr(manager, "redo_stack", None)
    if not isinstance(undo, list):
        return manager, None, None
    return manager, undo, redo if isinstance(redo, list) else None


def large_system_viewer_installed(mw) -> bool:
    styles = getattr(getattr(mw, "plugin_manager", None), "custom_3d_styles", None)
    return isinstance(styles, dict) and LSV_STYLE in styles


def open_with_host(mw, path: str) -> bool:
    """Open a file the way File > Open / the command line does: plugin openers first
    (the Large System Viewer takes PDB/GRO/XYZ when installed), then the host."""
    init = getattr(mw, "init_manager", None)
    loader = getattr(init, "load_command_line_file", None)
    if loader is None:
        return False
    loader(path)
    return True


class CellKeeper:
    def __init__(self, context):
        self.context = context
        self.overlay_enabled = True
        self.actor_names: list[str] = []
        self._drawn_signature = None
        self._last_mol = None
        self._pending: dict | None = None  # {"cell": json, "n_atoms": n} from a project
        self._observer = None
        self._observed = None
        self._scheduled = False
        self.listeners = []  # callables run when the current cell may have changed

    # -- plotter hook -----------------------------------------------------

    def _plotter(self):
        try:
            return self.context.plotter
        except Exception:  # noqa: BLE001
            return None

    def install(self) -> None:
        plotter = self._plotter()
        window = None
        for attr in ("render_window", "ren_win"):
            try:
                window = getattr(plotter, attr, None)
            except Exception:  # noqa: BLE001 - pyvista raises when closed
                window = None
            if window is not None:
                break
        if window is None or window is self._observed:
            return
        try:
            self._observer = window.AddObserver("EndEvent", self._on_render)
            self._observed = window
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not watch the render window: %s", exc)

    def _on_render(self, *_args) -> None:
        if self._scheduled:
            return
        self._scheduled = True
        try:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, self.sync)
        except Exception:  # noqa: BLE001
            self._scheduled = False

    # -- cell bookkeeping -------------------------------------------------

    def current_cell(self) -> PeriodicCell | None:
        return read_cell(self._current_mol())

    def _current_mol(self):
        try:
            return self.context.current_molecule
        except Exception:  # noqa: BLE001
            return None

    def remember_for_load(self, data) -> None:
        """Project load: the cell arrives before the molecule; apply it on arrival."""
        if isinstance(data, dict) and data.get("cell"):
            self._pending = {
                "cell": data["cell"],
                "n_atoms": int(data.get("n_atoms", -1)),
            }
            self._last_mol = None
        else:
            self._pending = None

    def state_for_save(self) -> dict:
        mol = self._current_mol()
        cell = read_cell(mol)
        if cell is None:
            return {}
        return {"cell": cell.to_json(), "n_atoms": mol.GetNumAtoms()}

    def checkpoint(self) -> None:
        """Push an undo step for the current molecule and tag it with its cell.

        A change that only touches the cell leaves the host's snapshot identical (the
        property is not in it), so the host would not record a step; one is added here
        so that undo also reverts cell-only edits."""
        mw = _mw(self.context)
        manager, undo, redo = _stacks(mw)
        top_before = undo[-1] if undo else None
        try:
            self.context.push_undo_checkpoint()
        except Exception as exc:  # noqa: BLE001
            logger.debug("push_undo_checkpoint failed: %s", exc)
        if undo is None or not undo:
            return
        cell = read_cell(self._current_mol())
        value = None if cell is None else cell.to_json()
        top = undo[-1]
        if top is top_before and top.get(STATE_KEY) != value:
            state = copy.deepcopy(top)
            state[STATE_KEY] = value
            undo.append(state)
            if redo is not None:
                redo.clear()
            update = getattr(manager, "update_undo_redo_actions", None)
            if callable(update):
                update()
            try:
                self.context.mark_project_modified()
            except Exception:
                logger.debug("Ignored error", exc_info=True)
        else:
            top[STATE_KEY] = value
        self._last_mol = self._current_mol()
        self._notify()

    def _notify(self) -> None:
        for listener in list(self.listeners):
            try:
                listener()
            except Exception:
                logger.debug("Cell listener failed", exc_info=True)

    def sync(self) -> None:
        """Runs after each host render (and on demand)."""
        self._scheduled = False
        mol = self._current_mol()
        changed = mol is not self._last_mol
        if changed:
            self._last_mol = mol
            if mol is not None and not mol.HasProp(PROP):
                self._restore_cell(mol)
            self._tag_top_state(mol)
            self._notify()
        self.draw_overlay()

    def _restore_cell(self, mol) -> None:
        n = mol.GetNumAtoms()
        if self._pending is not None and self._pending.get("n_atoms") in (n, -1):
            mol.SetProp(PROP, self._pending["cell"])
            self._pending = None
            return
        _, undo, _ = _stacks(_mw(self.context))
        if not undo:
            return
        top = undo[-1]
        value = top.get(STATE_KEY)
        ids = top.get("mol_3d_atom_ids")
        if value and (ids is None or len(ids) == n):
            mol.SetProp(PROP, value)

    def _tag_top_state(self, mol) -> None:
        """A state the host pushed itself (an edit after a build) carries no tag; it
        belongs to the current molecule, so tag it with that molecule's cell."""
        _, undo, _ = _stacks(_mw(self.context))
        if not undo or mol is None:
            return
        top = undo[-1]
        if STATE_KEY in top:
            return
        ids = top.get("mol_3d_atom_ids")
        if ids is not None and len(ids) != mol.GetNumAtoms():
            return
        cell = read_cell(mol)
        top[STATE_KEY] = None if cell is None else cell.to_json()

    # -- overlay ----------------------------------------------------------

    def set_overlay(self, enabled: bool) -> None:
        self.overlay_enabled = bool(enabled)
        self._drawn_signature = None
        self.draw_overlay()

    def clear_overlay(self) -> None:
        plotter = self._plotter()
        if plotter is not None:
            for name in self.actor_names:
                try:
                    plotter.remove_actor(name, render=False)
                except Exception:
                    logger.debug("Ignored error", exc_info=True)
        self.actor_names = []
        self._drawn_signature = None

    def draw_overlay(self) -> None:
        plotter = self._plotter()
        if plotter is None:
            return
        cell = self.current_cell() if self.overlay_enabled else None
        signature = None if cell is None else cell.to_json()
        actors = getattr(plotter, "actors", {}) or {}
        present = bool(self.actor_names) and all(n in actors for n in self.actor_names)
        if signature == self._drawn_signature and (present or signature is None):
            return
        self.clear_overlay()
        if cell is None:
            self._render(plotter)
            return
        from .cell_preview import cell_segments

        names = []
        for k, (start, end, colour, label) in enumerate(cell_segments(cell.lattice)):
            name = f"{ACTOR_PREFIX}_{k}"
            try:
                plotter.add_lines(
                    np.array([start, end]) + cell.origin,
                    color=colour,
                    width=3 if label else 2,
                    name=name,
                )
                names.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not draw the cell edge: %s", exc)
        self.actor_names = names
        self._drawn_signature = signature
        self._render(plotter)

    @staticmethod
    def _render(plotter) -> None:
        try:
            plotter.render()
        except Exception:
            logger.debug("Ignored error", exc_info=True)
