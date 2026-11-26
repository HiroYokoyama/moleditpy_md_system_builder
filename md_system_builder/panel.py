"""The MD System Builder dock: Box, Pack, Solvate and Cell tools tabs.

All four tabs end in :meth:`BuilderPanel.deliver`, the single output path: the result
becomes the host's molecule as one undoable step, or (large systems, or on request)
is written to PDB/GRO/XYZ and opened through the host's normal open path.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import celltools, molio, packing, protein, solvate
from .cellmeta import PeriodicCell, bounding_box_cell, read_cell, write_cell
from .host_link import (
    LSV_HOST_LINK_LIMIT,
    LSV_SWITCH_THRESHOLD,
    large_system_viewer_installed,
    open_with_host,
)
from .system import System, from_rdkit, to_rdkit
from .worker import Job

logger = logging.getLogger(__name__)

TITLE = "MD System Builder"

BOX_LENGTHS = "Lengths and angles"
BOX_VECTORS = "Cell vectors"
BOX_PADDING = "Bounding box + padding"
BOX_CIF = "From the CIF Viewer panel"

SOLV_BOX_CELL = "Existing cell"
SOLV_BOX_PADDING = "Bounding box + padding"
SOLV_BOX_LENGTHS = "Explicit lengths"
SOLVENT_WATER = "TIP3P water (pre-equilibrated box)"
SOLVENT_SMILES = "Molecule (SMILES) at a density"

REGION_LABELS = {
    packing.REGION_BOX: "Anywhere in the cell",
    packing.REGION_INSIDE_SPHERE: "Inside a sphere",
    packing.REGION_OUTSIDE_SPHERE: "Outside a sphere",
    packing.REGION_INSIDE_SLAB: "Inside a slab",
    packing.REGION_OUTSIDE_SLAB: "Outside a slab",
}


def default_settings() -> dict:
    return {
        "tolerance": 2.0,
        "seed": 12345,
        "max_attempts": 2000,
        "padding": 10.0,
        "solute_cutoff": 2.4,
        "salt": "NaCl",
        "neutralize": True,
        "concentration": 0.15,
        "ion_distance": 5.0,
        "overlay": True,
        "file_threshold": LSV_HOST_LINK_LIMIT,
        "output_format": "pdb",
        "histidine": "HIE",
        "remove_water": True,
        "remove_other": False,
    }


def _spin(lo, hi, value, decimals=3, step=0.1, suffix=""):
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    return box


def _ispin(lo, hi, value):
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    return box


class BuilderPanel(QWidget):
    def __init__(
        self, context, keeper, settings: dict, get_cif_viewer=None, parent=None
    ):
        super().__init__(parent)
        self.context = context
        self.keeper = keeper
        self.settings = settings
        self.get_cif_viewer = get_cif_viewer
        self.components: list[packing.Component] = []
        self.job: Job | None = None
        self._job_label = ""
        self._deliver_after = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        self.cell_label = QLabel()
        self.cell_label.setWordWrap(True)
        outer.addWidget(self.cell_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll(self._build_protein_tab()), "Protein")
        self.tabs.addTab(self._scroll(self._build_box_tab()), "Box")
        self.tabs.addTab(self._scroll(self._build_pack_tab()), "Pack")
        self.tabs.addTab(self._scroll(self._build_solvate_tab()), "Solvate")
        self.tabs.addTab(self._scroll(self._build_cell_tab()), "Cell tools")
        outer.addWidget(self.tabs, 1)

        out_box = QGroupBox("Output")
        out = QFormLayout(out_box)
        self.write_file_check = QCheckBox(
            "Write a file and open it (instead of replacing the molecule)"
        )
        out.addRow(self.write_file_check)
        self.format_combo = QComboBox()
        self.format_combo.addItems(["pdb", "gro", "xyz"])
        out.addRow("File format:", self.format_combo)
        self.threshold_spin = _ispin(
            100, 100_000_000, int(settings.get("file_threshold", LSV_HOST_LINK_LIMIT))
        )
        self.threshold_spin.setToolTip(
            "Results with more atoms than this are always written to a file and opened "
            "through File > Open (the Large System Viewer's view-only size by default)."
        )
        out.addRow("Always write a file above (atoms):", self.threshold_spin)
        self.overlay_check = QCheckBox("Show the cell box in the 3D view")
        self.overlay_check.setChecked(bool(settings.get("overlay", True)))
        self.overlay_check.toggled.connect(self._overlay_toggled)
        out.addRow(self.overlay_check)
        outer.addWidget(out_box)

        bar = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        bar.addWidget(self.progress, 1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_job)
        bar.addWidget(self.cancel_button)
        outer.addLayout(bar)
        self.status = QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)

        self.apply_settings(settings)
        self.keeper.listeners.append(self.refresh)
        self.refresh()

    @staticmethod
    def _scroll(widget):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    # ------------------------------------------------------------------
    # Protein tab
    # ------------------------------------------------------------------

    def _build_protein_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel(
            "Prepares a protein opened from a PDB file for MD with the Amber ff14SB "
            "force field: every hydrogen is rebuilt for pH 7 (Lys, Arg, Asp and Glu "
            "charged), chain ends are charged (OXT is added when missing), cysteines "
            "closer than 2.5 A are joined as disulfides and missing heavy atoms are "
            "added. Run it before Solvate."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        box = QGroupBox("Options")
        form = QFormLayout(box)
        self.histidine_combo = QComboBox()
        for key, label in (
            ("HIE", "HIE (H on NE2)"),
            ("HID", "HID (H on ND1)"),
            ("HIP", "HIP (both, charged)"),
        ):
            self.histidine_combo.addItem(label, key)
        self.histidine_combo.setToolTip(
            "Used for histidines without hydrogens in the file; residues named HID, "
            "HIE or HIP, or with ring hydrogens, keep their state."
        )
        form.addRow("Histidine:", self.histidine_combo)
        self.remove_water_check = QCheckBox("Remove crystal waters")
        form.addRow(self.remove_water_check)
        self.remove_other_check = QCheckBox(
            "Remove other groups (ions, ligands, additives)"
        )
        self.remove_other_check.setToolTip(
            "Kept groups stay as they are: they need hydrogens and parameters of "
            "their own (Force Field Builder uses Sage for them)."
        )
        form.addRow(self.remove_other_check)
        layout.addWidget(box)
        run = QPushButton("Prepare Protein")
        run.clicked.connect(self.run_protein)
        layout.addWidget(run)
        self.protein_report = QLabel()
        self.protein_report.setWordWrap(True)
        self.protein_report.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.protein_report)
        layout.addStretch(1)
        return page

    def run_protein(self):
        mol = self.current_mol()
        if mol is None or not mol.GetNumAtoms():
            self.error("Open a protein (PDB file) first.")
            return
        try:
            source = from_rdkit(mol)
        except ValueError as exc:
            self.error(str(exc))
            return
        histidine = self.histidine_combo.currentData()
        remove_water = self.remove_water_check.isChecked()
        remove_other = self.remove_other_check.isChecked()
        self.protein_report.setText("")

        def job(progress, cancelled):
            result, report = protein.prepare(
                source, histidine, remove_water, remove_other
            )
            result.note = " ".join(report.lines())
            result.report_lines = report.lines()
            return result

        self.start_job("Preparing the protein", job)

    # ------------------------------------------------------------------
    # Box tab
    # ------------------------------------------------------------------

    def _build_box_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.box_mode = QComboBox()
        self.box_mode.addItems([BOX_LENGTHS, BOX_VECTORS, BOX_PADDING, BOX_CIF])
        layout.addWidget(self.box_mode)
        self.box_stack = QStackedWidget()

        lengths = QWidget()
        grid = QGridLayout(lengths)
        self.len_spins = [
            _spin(0.1, 10000, 30.0, step=1.0, suffix=" A") for _ in range(3)
        ]
        self.ang_spins = [
            _spin(10, 170, 90.0, decimals=2, step=1.0, suffix=" deg") for _ in range(3)
        ]
        for k, label in enumerate("abc"):
            grid.addWidget(QLabel(label), 0, k)
            grid.addWidget(self.len_spins[k], 1, k)
        for k, label in enumerate(("alpha", "beta", "gamma")):
            grid.addWidget(QLabel(label), 2, k)
            grid.addWidget(self.ang_spins[k], 3, k)
        self.box_stack.addWidget(lengths)

        vectors = QWidget()
        vgrid = QGridLayout(vectors)
        self.vec_spins = [
            [_spin(-10000, 10000, 30.0 if i == j else 0.0, step=1.0) for j in range(3)]
            for i in range(3)
        ]
        for i, label in enumerate("abc"):
            vgrid.addWidget(QLabel(label), i, 0)
            for j in range(3):
                vgrid.addWidget(self.vec_spins[i][j], i, j + 1)
        self.box_stack.addWidget(vectors)

        padding = QWidget()
        pform = QFormLayout(padding)
        self.box_padding = _spin(0, 1000, 10.0, step=1.0, suffix=" A")
        pform.addRow("Padding:", self.box_padding)
        self.box_cubic = QCheckBox("Cubic")
        pform.addRow(self.box_cubic)
        self.box_stack.addWidget(padding)

        cif = QLabel("Takes the cell of the structure open in the CIF Viewer panel.")
        cif.setWordWrap(True)
        self.box_stack.addWidget(cif)
        layout.addWidget(self.box_stack)
        self.box_mode.currentIndexChanged.connect(self.box_stack.setCurrentIndex)

        self.box_centre = QCheckBox("Move the atoms to the centre of the new cell")
        self.box_centre.setChecked(True)
        layout.addWidget(self.box_centre)
        row = QHBoxLayout()
        read = QPushButton("Read current cell")
        read.clicked.connect(self.read_current_cell)
        apply = QPushButton("Set cell")
        apply.clicked.connect(self.set_box)
        clear = QPushButton("Clear cell")
        clear.clicked.connect(self.clear_box)
        for button in (read, apply, clear):
            row.addWidget(button)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def box_cell(self, xyz=None) -> PeriodicCell:
        mode = self.box_mode.currentText()
        if mode == BOX_LENGTHS:
            return PeriodicCell.from_parameters(
                [s.value() for s in self.len_spins], [s.value() for s in self.ang_spins]
            )
        if mode == BOX_VECTORS:
            return PeriodicCell(
                np.array([[s.value() for s in row] for row in self.vec_spins])
            )
        if mode == BOX_PADDING:
            if xyz is None:
                raise ValueError("There is no molecule to measure.")
            return bounding_box_cell(
                xyz, self.box_padding.value(), self.box_cubic.isChecked()
            )
        widget = self.get_cif_viewer() if self.get_cif_viewer is not None else None
        if widget is None:
            raise ValueError(
                "The CIF Viewer panel is not open (View > CIF Viewer Panel)."
            )
        from .cell_model import cell_from_viewer_structure

        cif = cell_from_viewer_structure(
            getattr(widget, "structure", None), expand_asymmetric=False
        )
        return PeriodicCell(np.asarray(cif.lattice, dtype=float))

    def read_current_cell(self):
        cell = self.keeper.current_cell()
        if cell is None:
            self.say("The current molecule has no cell.")
            return
        (a, b, c), angles = cell.parameters
        for spin, value in zip(self.len_spins, (a, b, c)):
            spin.setValue(value)
        for spin, value in zip(self.ang_spins, angles):
            spin.setValue(value)
        for i in range(3):
            for j in range(3):
                self.vec_spins[i][j].setValue(float(cell.lattice[i, j]))

    def set_box(self):
        mol = self.current_mol()
        try:
            if mol is None or mol.GetNumConformers() == 0:
                raise ValueError("Build or load a 3D molecule first.")
            xyz = np.array(mol.GetConformer().GetPositions())
            cell = self.box_cell(xyz)
            if (
                self.box_mode.currentText() != BOX_PADDING
                and self.box_centre.isChecked()
            ):
                xyz = xyz + (cell.centre - xyz.mean(axis=0))
        except ValueError as exc:
            self.error(str(exc))
            return
        self.apply_in_place(xyz, cell, "Cell set: " + cell.describe())

    def clear_box(self):
        mol = self.current_mol()
        if mol is None or read_cell(mol) is None:
            self.say("The current molecule has no cell.")
            return
        self.apply_in_place(None, None, "Cell cleared.")

    # ------------------------------------------------------------------
    # Pack tab
    # ------------------------------------------------------------------

    def _build_pack_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.comp_table = QTableWidget(0, 4)
        self.comp_table.setHorizontalHeaderLabels(["Name", "Atoms", "Count", "Region"])
        self.comp_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.comp_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.comp_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.comp_table.setMinimumHeight(140)
        self.comp_table.itemChanged.connect(self._component_edited)
        self.comp_table.itemSelectionChanged.connect(self._load_region)
        layout.addWidget(self.comp_table)
        row = QHBoxLayout()
        for text, slot in (
            ("Current molecule", self.add_current_component),
            ("SMILES...", self.add_smiles_component),
            ("File...", self.add_file_component),
            ("Remove", self.remove_component),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        region_box = QGroupBox("Region of the selected component")
        rform = QFormLayout(region_box)
        self.region_kind = QComboBox()
        for kind in packing.REGIONS:
            self.region_kind.addItem(REGION_LABELS[kind], kind)
        rform.addRow("Region:", self.region_kind)
        self.region_cell_centre = QCheckBox("Centred on the cell")
        self.region_cell_centre.setChecked(True)
        rform.addRow(self.region_cell_centre)
        centre_row = QHBoxLayout()
        self.region_centre = [_spin(-10000, 10000, 0.0, step=1.0) for _ in range(3)]
        for spin in self.region_centre:
            centre_row.addWidget(spin)
        rform.addRow("Centre (x y z):", centre_row)
        self.region_radius = _spin(0.1, 10000, 10.0, step=1.0, suffix=" A")
        rform.addRow("Sphere radius:", self.region_radius)
        self.region_axis = QComboBox()
        self.region_axis.addItems(["x", "y", "z"])
        self.region_axis.setCurrentIndex(2)
        rform.addRow("Slab axis:", self.region_axis)
        self.region_low = _spin(-10000, 10000, 0.0, step=1.0, suffix=" A")
        self.region_high = _spin(-10000, 10000, 10.0, step=1.0, suffix=" A")
        rform.addRow("Slab from:", self.region_low)
        rform.addRow("Slab to:", self.region_high)
        for widget in (self.region_kind, self.region_axis):
            widget.currentIndexChanged.connect(self._store_region)
        for widget in [
            self.region_radius,
            self.region_low,
            self.region_high,
        ] + self.region_centre:
            widget.valueChanged.connect(self._store_region)
        self.region_cell_centre.toggled.connect(self._store_region)
        layout.addWidget(region_box)

        box = QGroupBox("Cell and amounts")
        form = QFormLayout(box)
        self.pack_cell_mode = QComboBox()
        self.pack_cell_mode.addItems(["Current cell", "Box (a b c below)"])
        form.addRow("Cell:", self.pack_cell_mode)
        lens = QHBoxLayout()
        self.pack_lengths = [_spin(1, 10000, 40.0, step=1.0) for _ in range(3)]
        for spin in self.pack_lengths:
            lens.addWidget(spin)
        form.addRow("a b c (A):", lens)
        self.pack_use_density = QCheckBox(
            "Counts from a density (the Count column gives mole ratios)"
        )
        form.addRow(self.pack_use_density)
        self.pack_density = _spin(0.01, 30, 1.0, step=0.05, suffix=" g/cm3")
        form.addRow("Density:", self.pack_density)
        self.pack_fixed = QCheckBox(
            "Keep the current molecule as a fixed solute, centred"
        )
        form.addRow(self.pack_fixed)
        self.pack_tolerance = _spin(0.5, 10, 2.0, step=0.1, suffix=" A")
        form.addRow("Minimum distance:", self.pack_tolerance)
        self.pack_attempts = _ispin(10, 1_000_000, 2000)
        form.addRow("Random trials per molecule:", self.pack_attempts)
        self.pack_seed = _ispin(0, 2_147_483_647, 12345)
        form.addRow("Seed:", self.pack_seed)
        layout.addWidget(box)
        run = QPushButton("Pack")
        run.clicked.connect(self.run_pack)
        layout.addWidget(run)
        layout.addStretch(1)
        return page

    def _add_component(self, template: System, name: str, count: int = 10):
        template = template.copy()
        if template.n_molecules > 1:
            # A multi-fragment source would be packed as one rigid unit; say so.
            self.say(
                f"{name}: {template.n_molecules} fragments are packed together as one unit."
            )
        comp = packing.Component(template, count, name[:4] or "MOL")
        self.components.append(comp)
        self._refresh_components()
        self.comp_table.selectRow(len(self.components) - 1)

    def add_current_component(self):
        mol = self.current_mol()
        try:
            if mol is None:
                raise ValueError("There is no current molecule.")
            template = from_rdkit(mol)
        except ValueError as exc:
            self.error(str(exc))
            return
        name = template.resnames[0] if template.resnames[0] != "MOL" else "MOL"
        self._add_component(template, name)

    def add_smiles_component(self):
        smiles, ok = QInputDialog.getText(self, TITLE, "SMILES:")
        if not ok or not smiles.strip():
            return
        name, ok = QInputDialog.getText(
            self, TITLE, "Residue name (up to 4 characters):", text="MOL"
        )
        if not ok:
            return
        try:
            template = molio.from_smiles(
                smiles, name=(name or "MOL").strip()[:4] or "MOL"
            )
        except ValueError as exc:
            self.error(str(exc))
            return
        self._add_component(template, (name or "MOL").strip()[:4] or "MOL")

    def add_file_component(self):
        path, _ = QFileDialog.getOpenFileName(self, TITLE, "", molio.FILE_FILTER)
        if not path:
            return
        try:
            template = molio.from_file(path)
        except ValueError as exc:
            self.error(str(exc))
            return
        self._add_component(template, template.resnames[0])

    def remove_component(self):
        row = self.comp_table.currentRow()
        if 0 <= row < len(self.components):
            del self.components[row]
            self._refresh_components()

    def _refresh_components(self):
        self.comp_table.blockSignals(True)
        try:
            self.comp_table.setRowCount(len(self.components))
            for row, comp in enumerate(self.components):
                name = QTableWidgetItem(comp.name)
                atoms = QTableWidgetItem(str(comp.template.n_atoms))
                atoms.setFlags(atoms.flags() & ~Qt.ItemFlag.ItemIsEditable)
                count = QTableWidgetItem(str(comp.count))
                region = QTableWidgetItem(REGION_LABELS[comp.region.kind])
                region.setFlags(region.flags() & ~Qt.ItemFlag.ItemIsEditable)
                for col, item in enumerate((name, atoms, count, region)):
                    self.comp_table.setItem(row, col, item)
        finally:
            self.comp_table.blockSignals(False)

    def _component_edited(self, item):
        row, col = item.row(), item.column()
        if not 0 <= row < len(self.components):
            return
        comp = self.components[row]
        if col == 0:
            comp.name = item.text().strip()[:4] or "MOL"
        elif col == 2:
            try:
                comp.count = max(0, int(item.text()))
            except ValueError:
                pass
        self._refresh_components()

    def _selected_component(self):
        row = self.comp_table.currentRow()
        return self.components[row] if 0 <= row < len(self.components) else None

    def _load_region(self):
        comp = self._selected_component()
        if comp is None:
            return
        reg = comp.region
        widgets = [
            self.region_kind,
            self.region_axis,
            self.region_radius,
            self.region_low,
            self.region_high,
            self.region_cell_centre,
        ] + self.region_centre
        for w in widgets:
            w.blockSignals(True)
        try:
            self.region_kind.setCurrentIndex(packing.REGIONS.index(reg.kind))
            self.region_axis.setCurrentIndex(int(reg.axis))
            self.region_radius.setValue(reg.radius)
            self.region_low.setValue(reg.low)
            self.region_high.setValue(reg.high)
            self.region_cell_centre.setChecked(reg.centre is None)
            for spin, value in zip(self.region_centre, reg.centre or (0.0, 0.0, 0.0)):
                spin.setValue(float(value))
        finally:
            for w in widgets:
                w.blockSignals(False)

    def _store_region(self, *_args):
        comp = self._selected_component()
        if comp is None:
            return
        try:
            comp.region = packing.Region(
                kind=self.region_kind.currentData(),
                centre=None
                if self.region_cell_centre.isChecked()
                else [s.value() for s in self.region_centre],
                radius=self.region_radius.value(),
                axis=self.region_axis.currentIndex(),
                low=self.region_low.value(),
                high=self.region_high.value(),
            )
        except ValueError as exc:
            self.say(str(exc))
            return
        self._refresh_components()

    def pack_cell(self) -> PeriodicCell:
        if self.pack_cell_mode.currentIndex() == 0:
            cell = self.keeper.current_cell()
            if cell is None:
                raise ValueError(
                    "The current molecule has no cell. Set one on the Box tab or choose 'Box'."
                )
            return cell
        return PeriodicCell.from_lengths([s.value() for s in self.pack_lengths])

    def run_pack(self):
        try:
            if not self.components:
                raise ValueError("Add at least one component.")
            cell = self.pack_cell()
            fixed = None
            if self.pack_fixed.isChecked():
                mol = self.current_mol()
                if mol is None:
                    raise ValueError(
                        "There is no current molecule to keep as the solute."
                    )
                fixed = from_rdkit(mol)
            comps = [
                packing.Component(c.template, c.count, c.name, c.region)
                for c in self.components
            ]
            if self.pack_use_density.isChecked():
                volume = packing.free_volume(cell, fixed)
                counts = packing.mixture_counts(
                    self.pack_density.value(), volume, comps, [c.count for c in comps]
                )
                for comp, n in zip(comps, counts):
                    comp.count = n
            if sum(c.count for c in comps) == 0:
                raise ValueError("Nothing to pack: every count is zero.")
        except ValueError as exc:
            self.error(str(exc))
            return
        tolerance = self.pack_tolerance.value()
        seed = self.pack_seed.value()
        attempts = self.pack_attempts.value()

        def job(progress, cancelled):
            return packing.pack(
                cell,
                comps,
                tolerance=tolerance,
                seed=seed,
                fixed=fixed,
                centre_fixed=True,
                max_attempts=attempts,
                progress=progress,
                cancelled=cancelled,
            ).system

        summary = ", ".join(f"{c.count} {c.name}" for c in comps)
        self.start_job("Packing " + summary, job)

    # ------------------------------------------------------------------
    # Solvate tab
    # ------------------------------------------------------------------

    def _build_solvate_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("Box")
        form = QFormLayout(box)
        self.solv_box_mode = QComboBox()
        self.solv_box_mode.addItems([SOLV_BOX_PADDING, SOLV_BOX_LENGTHS, SOLV_BOX_CELL])
        form.addRow("Box:", self.solv_box_mode)
        self.solv_padding = _spin(0, 1000, 10.0, step=1.0, suffix=" A")
        form.addRow("Padding:", self.solv_padding)
        self.solv_cubic = QCheckBox("Cubic")
        self.solv_cubic.setChecked(True)
        form.addRow(self.solv_cubic)
        lens = QHBoxLayout()
        self.solv_lengths = [_spin(1, 10000, 40.0, step=1.0) for _ in range(3)]
        for spin in self.solv_lengths:
            lens.addWidget(spin)
        form.addRow("a b c (A):", lens)
        layout.addWidget(box)

        sbox = QGroupBox("Solvent")
        sform = QFormLayout(sbox)
        self.solvent_mode = QComboBox()
        self.solvent_mode.addItems([SOLVENT_WATER, SOLVENT_SMILES])
        sform.addRow(self.solvent_mode)
        self.solvent_smiles = QLineEdit("CO")
        sform.addRow("SMILES:", self.solvent_smiles)
        self.solvent_name = QLineEdit("MOH")
        sform.addRow("Residue name:", self.solvent_name)
        self.solvent_density = _spin(0.01, 30, 0.79, step=0.05, suffix=" g/cm3")
        sform.addRow("Density:", self.solvent_density)
        self.solute_cutoff = _spin(0.5, 10, 2.4, step=0.1, suffix=" A")
        sform.addRow("Remove solvent within:", self.solute_cutoff)
        layout.addWidget(sbox)
        self.solvent_mode.currentIndexChanged.connect(self._solvent_mode_changed)

        ibox = QGroupBox("Ions (water only)")
        iform = QFormLayout(ibox)
        self.salt_combo = QComboBox()
        self.salt_combo.addItems(list(solvate.SALTS))
        iform.addRow("Salt:", self.salt_combo)
        self.neutralize_check = QCheckBox("Neutralise the solute charge")
        self.neutralize_check.setChecked(True)
        iform.addRow(self.neutralize_check)
        self.concentration = _spin(0, 5, 0.15, decimals=3, step=0.05, suffix=" mol/L")
        iform.addRow("Salt concentration:", self.concentration)
        self.ion_distance = _spin(0, 30, 5.0, step=0.5, suffix=" A")
        iform.addRow("Minimum ion-solute distance:", self.ion_distance)
        self.solv_seed = _ispin(0, 2_147_483_647, 12345)
        iform.addRow("Seed:", self.solv_seed)
        layout.addWidget(ibox)
        self.ion_box = ibox
        self._solvent_mode_changed()
        run = QPushButton("Solvate")
        run.clicked.connect(self.run_solvate)
        layout.addWidget(run)
        layout.addStretch(1)
        return page

    def _solvent_mode_changed(self, *_args):
        water = self.solvent_mode.currentText() == SOLVENT_WATER
        for widget in (self.solvent_smiles, self.solvent_name, self.solvent_density):
            widget.setEnabled(not water)
        if hasattr(self, "ion_box"):
            self.ion_box.setEnabled(water)

    def run_solvate(self):
        mol = self.current_mol()
        try:
            solute = from_rdkit(mol) if mol is not None and mol.GetNumAtoms() else None
            mode = self.solv_box_mode.currentText()
            if mode == SOLV_BOX_CELL:
                cell = self.keeper.current_cell()
                if cell is None:
                    raise ValueError("The current molecule has no cell.")
            elif mode == SOLV_BOX_PADDING:
                if solute is None:
                    raise ValueError(
                        "Padding needs a solute; use explicit lengths for pure solvent."
                    )
                cell = bounding_box_cell(
                    solute.xyz, self.solv_padding.value(), self.solv_cubic.isChecked()
                )
            else:
                cell = PeriodicCell.from_lengths([s.value() for s in self.solv_lengths])
                if solute is not None:
                    solute.xyz = solute.xyz - solute.xyz.mean(axis=0) + cell.centre
            water = self.solvent_mode.currentText() == SOLVENT_WATER
            solvent = None
            if not water:
                name = self.solvent_name.text().strip()[:4] or "SOL"
                solvent = molio.from_smiles(self.solvent_smiles.text(), name=name)
        except ValueError as exc:
            self.error(str(exc))
            return
        cutoff = self.solute_cutoff.value()
        salt = self.salt_combo.currentText()
        neutral = self.neutralize_check.isChecked()
        conc = self.concentration.value()
        ion_dist = self.ion_distance.value()
        seed = self.solv_seed.value()
        density = self.solvent_density.value()

        def job(progress, cancelled):
            if water:
                waters = solvate.tile_water(
                    cell, solute, cutoff, progress=progress, cancelled=cancelled
                )
                plan = solvate.plan_ions(
                    solvate.charge_of(solute), waters.n_molecules, salt, neutral, conc
                )
                body = solvate.place_ions(
                    waters, plan, solute, cell, ion_dist, seed=seed
                )
                note = f"{waters.n_molecules - plan.n_cation - plan.n_anion} waters, {plan.describe()}"
            else:
                body = solvate.pack_solvent(
                    cell,
                    solvent,
                    density,
                    solute,
                    tolerance=max(cutoff, 0.5),
                    seed=seed,
                    name=solvent.resnames[0],
                    progress=progress,
                    cancelled=cancelled,
                )
                note = f"{body.n_molecules} solvent molecules"
            parts = [solute] if solute is not None else []
            from .system import concatenate

            result = concatenate(parts + [body], cell=cell.copy())
            result.note = note
            return result

        self.start_job("Solvating", job)

    # ------------------------------------------------------------------
    # Cell tools tab
    # ------------------------------------------------------------------

    def _build_cell_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        for text, slot in (
            ("Centre in cell", self.tool_centre),
            ("Wrap atoms", self.tool_wrap_atoms),
            ("Wrap molecules", self.tool_wrap_molecules),
            ("Unwrap", self.tool_unwrap),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        sbox = QGroupBox("Supercell")
        srow = QHBoxLayout(sbox)
        self.super_spins = [_ispin(1, 50, 2) for _ in range(3)]
        for label, spin in zip("abc", self.super_spins):
            srow.addWidget(QLabel(f"n_{label}"))
            srow.addWidget(spin)
        button = QPushButton("Build")
        button.clicked.connect(self.tool_supercell)
        srow.addWidget(button)
        layout.addWidget(sbox)

        cbox = QGroupBox("Scale the cell")
        cform = QFormLayout(cbox)
        frow = QHBoxLayout()
        self.scale_spins = [
            _spin(0.1, 10, 1.0, decimals=4, step=0.01) for _ in range(3)
        ]
        for spin in self.scale_spins:
            frow.addWidget(spin)
        cform.addRow("Factors a b c:", frow)
        self.scale_mode = QComboBox()
        self.scale_mode.addItem("Move molecules rigidly with the cell", "molecules")
        self.scale_mode.addItem("Scale every atom position", "atoms")
        self.scale_mode.addItem("Cell only (coordinates unchanged)", "none")
        cform.addRow("Coordinates:", self.scale_mode)
        button = QPushButton("Scale")
        button.clicked.connect(self.tool_scale)
        cform.addRow(button)
        layout.addWidget(cbox)

        button = QPushButton("Convert to an orthorhombic cell")
        button.setToolTip(
            "Finds an equivalent orthorhombic (super)cell when one exists, e.g. hexagonal -> 2x."
        )
        button.clicked.connect(self.tool_orthorhombic)
        layout.addWidget(button)
        layout.addStretch(1)
        return page

    def _current_system(self) -> System:
        mol = self.current_mol()
        if mol is None:
            raise ValueError("There is no current molecule.")
        system = from_rdkit(mol)
        if system.cell is None:
            raise ValueError(
                "The current molecule has no cell. Set one on the Box tab first."
            )
        return system

    def _same_atoms_tool(self, func, message):
        try:
            system = self._current_system()
            result = func(system)
        except ValueError as exc:
            self.error(str(exc))
            return
        self.apply_in_place(result.xyz, result.cell, message)

    def tool_centre(self):
        self._same_atoms_tool(celltools.centre, "Centred in the cell.")

    def tool_wrap_atoms(self):
        self._same_atoms_tool(celltools.wrap_atoms, "Atoms wrapped into the cell.")

    def tool_wrap_molecules(self):
        self._same_atoms_tool(
            celltools.wrap_molecules, "Molecules wrapped into the cell."
        )

    def tool_unwrap(self):
        self._same_atoms_tool(celltools.unwrap, "Molecules made whole.")

    def tool_scale(self):
        factors = [s.value() for s in self.scale_spins]
        mode = self.scale_mode.currentData()
        self._same_atoms_tool(
            lambda s: celltools.scale(s, factors, mode), "Cell scaled."
        )

    def tool_supercell(self):
        reps = [s.value() for s in self.super_spins]
        try:
            system = self._current_system()
        except ValueError as exc:
            self.error(str(exc))
            return
        self.start_job(
            f"Supercell {reps[0]}x{reps[1]}x{reps[2]}",
            lambda p, c: celltools.supercell(system, reps),
        )

    def tool_orthorhombic(self):
        try:
            system = self._current_system()
            result = celltools.to_orthorhombic(system)
        except ValueError as exc:
            self.error(str(exc))
            return
        if result.n_atoms == system.n_atoms:
            self.apply_in_place(
                result.xyz, result.cell, "Orthorhombic cell: " + result.cell.describe()
            )
        else:
            self.deliver(result, "Orthorhombic supercell")

    # ------------------------------------------------------------------
    # jobs and output
    # ------------------------------------------------------------------

    def start_job(self, label, job):
        if self.job is not None and self.job.is_running():
            self.error("Another job is still running.")
            return
        self._job_label = label
        self.job = Job(job, self)
        self.job.progress.connect(self._on_progress)
        self.job.finished.connect(self._on_finished)
        self.job.failed.connect(self._on_failed)
        self.job.cancelled.connect(self._on_cancelled)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.say(label + "...")
        self.job.start()

    def cancel_job(self):
        if self.job is not None:
            self.job.cancel()

    def _job_done(self):
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)

    def _on_progress(self, done, total):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)

    def _on_finished(self, system):
        self._job_done()
        lines = getattr(system, "report_lines", None)
        if lines:
            self.protein_report.setText("\n".join(lines))
        self.deliver(system, self._job_label)

    def _on_failed(self, message):
        self._job_done()
        self.error(message)

    def _on_cancelled(self):
        self._job_done()
        self.say("Cancelled.")

    def current_mol(self):
        try:
            return self.context.current_molecule
        except Exception:  # noqa: BLE001 - host API guard
            return None

    def apply_in_place(self, xyz, cell, message):
        """Same atoms, new coordinates and/or cell: keeps every host atom property."""
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        mol = self.current_mol()
        if mol is None:
            self.error("There is no current molecule.")
            return
        new = Chem.Mol(mol)
        if xyz is not None:
            conf = new.GetConformer()
            for k, (x, y, z) in enumerate(np.asarray(xyz, dtype=float)):
                conf.SetAtomPosition(k, Point3D(float(x), float(y), float(z)))
        write_cell(new, cell)
        self.context.current_molecule = new
        self.keeper.checkpoint()
        self.keeper.sync()
        self.say(message)

    def deliver(self, system: System, label: str = ""):
        """The single output path for every tab."""
        n = system.n_atoms
        note = getattr(system, "note", "")
        counts = ", ".join(f"{c} {name}" for name, c in system.residue_counts())
        summary = f"{label}: {n} atoms ({counts})" + (f"; {note}" if note else "")
        lsv = large_system_viewer_installed(self.context.get_main_window())
        if self.write_file_check.isChecked() or n > self.threshold_spin.value():
            fmt = self.format_combo.currentText()
            path, _ = QFileDialog.getSaveFileName(
                self,
                TITLE,
                f"system.{fmt}",
                f"{fmt.upper()} files (*.{fmt});;All files (*)",
            )
            if not path:
                self.say(summary + ". Not saved.")
                return
            if not os.path.splitext(path)[1]:
                path += "." + fmt
            try:
                molio.write_system(system, path)
            except (OSError, ValueError) as exc:
                self.error(f"Could not write the file:\n{exc}")
                return
            if lsv or path.lower().endswith(".xyz"):
                open_with_host(self.context.get_main_window(), path)
                self.say(summary + f". Wrote and opened {os.path.basename(path)}.")
            else:
                self.say(
                    summary
                    + f". Wrote {path}. MoleditPy cannot open PDB/GRO by itself: "
                    "install the Large System Viewer plugin to view it."
                )
            return
        mol = to_rdkit(system)
        try:
            self.context.clear_canvas(push_to_undo=False)
        except Exception:
            logger.debug("Ignored error", exc_info=True)
        self.context.current_molecule = mol
        try:
            self.context.enter_3d_viewer_mode()
            self.context.reset_3d_camera()
        except Exception:
            logger.debug("Ignored error", exc_info=True)
        self.keeper.checkpoint()
        self.keeper.sync()
        if n > LSV_SWITCH_THRESHOLD and not lsv:
            summary += (
                ". Large systems draw much faster with the Large System Viewer plugin "
                "(Plugin Installer)."
            )
        self.say(summary + ".")

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------

    def _overlay_toggled(self, on):
        self.settings["overlay"] = bool(on)
        self.keeper.set_overlay(on)

    def refresh(self):
        cell = self.keeper.current_cell()
        mol = self.current_mol()
        atoms = mol.GetNumAtoms() if mol is not None else 0
        self.cell_label.setText(
            f"Current molecule: {atoms} atoms. "
            + ("Cell: " + cell.describe() if cell is not None else "No cell.")
        )

    def say(self, text):
        self.status.setText(text)
        try:
            self.context.show_status_message(text, 8000)
        except Exception:
            logger.debug("Ignored error", exc_info=True)

    def error(self, text):
        self.status.setText(text)
        QMessageBox.warning(self, TITLE, text)

    def read_settings(self) -> dict:
        return {
            "tolerance": self.pack_tolerance.value(),
            "seed": self.pack_seed.value(),
            "max_attempts": self.pack_attempts.value(),
            "padding": self.solv_padding.value(),
            "solute_cutoff": self.solute_cutoff.value(),
            "salt": self.salt_combo.currentText(),
            "neutralize": self.neutralize_check.isChecked(),
            "concentration": self.concentration.value(),
            "ion_distance": self.ion_distance.value(),
            "overlay": self.overlay_check.isChecked(),
            "file_threshold": self.threshold_spin.value(),
            "output_format": self.format_combo.currentText(),
            "histidine": self.histidine_combo.currentData(),
            "remove_water": self.remove_water_check.isChecked(),
            "remove_other": self.remove_other_check.isChecked(),
        }

    def apply_settings(self, settings):
        s = dict(default_settings())
        s.update(settings or {})
        self.pack_tolerance.setValue(float(s["tolerance"]))
        self.pack_seed.setValue(int(s["seed"]))
        self.solv_seed.setValue(int(s["seed"]))
        self.pack_attempts.setValue(int(s["max_attempts"]))
        self.solv_padding.setValue(float(s["padding"]))
        self.box_padding.setValue(float(s["padding"]))
        self.solute_cutoff.setValue(float(s["solute_cutoff"]))
        index = self.salt_combo.findText(str(s["salt"]))
        if index >= 0:
            self.salt_combo.setCurrentIndex(index)
        self.neutralize_check.setChecked(bool(s["neutralize"]))
        self.concentration.setValue(float(s["concentration"]))
        self.ion_distance.setValue(float(s["ion_distance"]))
        self.overlay_check.setChecked(bool(s["overlay"]))
        self.threshold_spin.setValue(int(s["file_threshold"]))
        index = self.format_combo.findText(str(s["output_format"]))
        if index >= 0:
            self.format_combo.setCurrentIndex(index)
        index = self.histidine_combo.findData(str(s["histidine"]))
        if index >= 0:
            self.histidine_combo.setCurrentIndex(index)
        self.remove_water_check.setChecked(bool(s["remove_water"]))
        self.remove_other_check.setChecked(bool(s["remove_other"]))

    def shutdown(self):
        self.settings.update(self.read_settings())
        if self.job is not None and self.job.is_running():
            self.job.cancel()
            self.job.wait(3000)
        if self.refresh in self.keeper.listeners:
            self.keeper.listeners.remove(self.refresh)
