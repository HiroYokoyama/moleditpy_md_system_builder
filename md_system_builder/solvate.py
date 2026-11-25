"""Solvation: a pre-equilibrated TIP3P box tiled into the cell, or any solvent packed at
a density, then counter-ions and salt that replace waters far from the solute."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .cellmeta import PeriodicCell
from .packing import (
    Cancelled,
    CellGrid,
    Component,
    count_for_density,
    free_volume,
    pack,
)
from .system import System, concatenate

WATER_RESNAME = "SOL"
ION_TABLE = {
    # name: (element, formal charge, residue name)
    "Na+": ("Na", 1, "NA"),
    "K+": ("K", 1, "K"),
    "Cl-": ("Cl", -1, "CL"),
}
SALTS = {"NaCl": ("Na+", "Cl-"), "KCl": ("K+", "Cl-")}
WATER_PER_LITRE = 55.345  # mol/L of water at 298 K, 0.997 g/cm3


def water_template() -> System:
    """One TIP3P water (O, H, H) with bonds, centred on the oxygen."""
    from . import water_box

    xyz = _box_coordinates(water_box)[:3]
    xyz = xyz - xyz[0]
    return System(
        ["O", "H", "H"],
        xyz,
        [(0, 1, 1.0), (0, 2, 1.0)],
        resnames=[WATER_RESNAME] * 3,
        atom_names=["OW", "HW1", "HW2"],
    )


def _box_coordinates(module) -> np.ndarray:
    return np.array([[float(v) for v in row.split()] for row in module.COORDINATES])


def tile_water(
    cell: PeriodicCell,
    solute: System | None = None,
    solute_cutoff: float = 2.4,
    water_tolerance: float = 2.1,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> System:
    """Waters from the shipped TIP3P box, tiled over the cell.

    A water is kept when its oxygen falls inside the cell, none of its atoms is within
    ``solute_cutoff`` of a solute atom, and its oxygen is at least ``water_tolerance``
    from every oxygen already kept (that removes the clashes created where the tiled
    box meets itself across a periodic boundary that is not a multiple of its length).
    """
    from . import water_box

    box = float(water_box.BOX_LENGTH)
    sites = _box_coordinates(water_box).reshape(-1, 3, 3)
    # Range of tile indices that covers the cell's bounding box.
    corners = cell.to_cartesian(
        np.array(
            [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float
        )
    )
    lo = np.floor(corners.min(axis=0) / box).astype(int)
    hi = np.ceil(corners.max(axis=0) / box).astype(int)

    solute_grid = None
    if solute is not None and solute.n_atoms:
        solute_grid = CellGrid(cell, solute_cutoff)
        solute_grid.add(solute.xyz)
    water_grid = CellGrid(cell, water_tolerance)

    tiles = [
        (i, j, k)
        for i in range(lo[0], hi[0])
        for j in range(lo[1], hi[1])
        for k in range(lo[2], hi[2])
    ]
    kept = []
    for n_tile, (i, j, k) in enumerate(tiles):
        if cancelled is not None and cancelled():
            raise Cancelled("Cancelled.")
        shifted = sites + np.array([i, j, k], dtype=float) * box
        frac_o = cell.to_fractional(shifted[:, 0, :])
        inside = np.all((frac_o >= 0.0) & (frac_o < 1.0), axis=1)
        for mol in shifted[inside]:
            if solute_grid is not None and solute_grid.clashes(mol):
                continue
            if water_grid.clashes(mol[:1]):
                continue
            water_grid.add(mol[:1])
            kept.append(mol)
        if progress is not None:
            progress(n_tile + 1, len(tiles))

    n = len(kept)
    if n == 0:
        return System(cell=cell.copy())
    xyz = np.array(kept).reshape(-1, 3)
    return System(
        ["O", "H", "H"] * n,
        xyz,
        [(3 * m + a, 3 * m + b, 1.0) for m in range(n) for a, b in ((0, 1), (0, 2))],
        mol_index=np.repeat(np.arange(n), 3),
        resnames=[WATER_RESNAME] * (3 * n),
        resnums=np.repeat(np.arange(1, n + 1), 3),
        atom_names=["OW", "HW1", "HW2"] * n,
        cell=cell.copy(),
    )


def pack_solvent(
    cell: PeriodicCell,
    solvent: System,
    density: float,
    solute: System | None = None,
    tolerance: float = 2.0,
    seed: int | None = None,
    max_attempts: int = 2000,
    name: str = "SOL",
    progress=None,
    cancelled=None,
) -> System:
    """Any solvent molecule at ``density`` g/cm3 through the packing core.

    The count fills the volume the solute leaves free; molecules that do not fit
    after the retry limit are dropped rather than failing, and the result reports
    the count actually placed.
    """
    volume = free_volume(cell, solute)
    count = count_for_density(density, volume, solvent.total_mass())
    comp = Component(solvent, count, name)
    result = pack(
        cell,
        [comp],
        tolerance=tolerance,
        seed=seed,
        fixed=None,
        max_attempts=max_attempts,
        progress=progress,
        cancelled=cancelled,
        allow_partial=True,
        **({} if solute is None else {"fixed": solute, "centre_fixed": False}),
    )
    system = result.system
    if solute is not None and solute.n_atoms:
        system = system.subset(range(solute.n_atoms, system.n_atoms))
    system.cell = cell.copy()
    return system


@dataclass
class IonPlan:
    cation: str
    anion: str
    n_cation: int
    n_anion: int

    def describe(self) -> str:
        return f"{self.n_cation} {self.cation}, {self.n_anion} {self.anion}"


def plan_ions(
    solute_charge: int,
    n_water: int,
    salt: str = "NaCl",
    neutralize: bool = True,
    concentration: float = 0.0,
) -> IonPlan:
    """Ion counts: salt pairs from the molar ``concentration`` (relative to the water
    count) plus counter-ions to cancel ``solute_charge`` when ``neutralize``."""
    if salt not in SALTS:
        raise ValueError(f"Unknown salt {salt}.")
    cation, anion = SALTS[salt]
    pairs = round(max(concentration, 0.0) * n_water / WATER_PER_LITRE)
    n_cat = n_an = pairs
    if neutralize and solute_charge:
        if solute_charge > 0:
            n_an += solute_charge
        else:
            n_cat += -solute_charge
    return IonPlan(cation, anion, n_cat, n_an)


def place_ions(
    waters: System,
    plan: IonPlan,
    solute: System | None,
    cell: PeriodicCell,
    min_solute_distance: float = 5.0,
    min_ion_distance: float = 5.0,
    seed: int | None = None,
) -> System:
    """Replace randomly chosen waters by ions; returns waters + ions (ions last).

    Candidate waters have their oxygen at least ``min_solute_distance`` from the solute
    and ``min_ion_distance`` from ions already placed.  The distances are relaxed in
    steps when there are not enough candidates, so a small box still gets its ions.
    """
    wanted = [plan.cation] * plan.n_cation + [plan.anion] * plan.n_anion
    if not wanted:
        return waters
    n_water = waters.n_molecules
    if len(wanted) > n_water:
        raise ValueError(
            f"{len(wanted)} ions requested but there are only {n_water} waters to replace."
        )
    rng = np.random.default_rng(seed)
    molecules = waters.molecules()
    oxygens = np.array([waters.xyz[int(atoms[0])] for atoms in molecules])

    if solute is not None and solute.n_atoms:
        solute_grid = CellGrid(cell, max(min_solute_distance, 0.1))
        solute_grid.add(solute.xyz)
        dist = solute_grid.min_distance(oxygens)
    else:
        dist = np.full(len(oxygens), np.inf)

    chosen: list[int] = []
    solute_limit, ion_limit = float(min_solute_distance), float(min_ion_distance)
    order = rng.permutation(len(oxygens))
    while len(chosen) < len(wanted):
        ion_grid = CellGrid(cell, max(ion_limit, 0.1))
        if chosen:
            ion_grid.add(oxygens[chosen])
        for m in order:
            if len(chosen) == len(wanted):
                break
            if m in chosen or dist[m] < solute_limit:
                continue
            if ion_limit > 0 and chosen and ion_grid.clashes(oxygens[m]):
                continue
            chosen.append(int(m))
            ion_grid.add(oxygens[m])
        if len(chosen) < len(wanted):
            if solute_limit <= 0 and ion_limit <= 0:
                raise ValueError("Could not find waters to replace by ions.")
            solute_limit = max(0.0, solute_limit - 1.0)
            ion_limit = max(0.0, ion_limit - 1.0)

    keep = np.ones(n_water, dtype=bool)
    keep[chosen] = False
    kept_atoms = (
        np.concatenate([molecules[m] for m in range(n_water) if keep[m]])
        if keep.any()
        else np.zeros(0, dtype=int)
    )
    remaining = (
        waters.subset(kept_atoms) if len(kept_atoms) else System(cell=cell.copy())
    )

    ions = []
    for label, m in zip(wanted, chosen):
        element, charge, resname = ION_TABLE[label]
        ions.append(
            System(
                [element],
                oxygens[m].reshape(1, 3),
                [],
                charges=np.array([charge]),
                resnames=[resname],
                atom_names=[resname],
            )
        )
    # Cations first, then anions: identical molecules stay consecutive.
    result = concatenate([remaining] + ions, cell=cell.copy())
    return result


def solvent_count_for_density(
    cell: PeriodicCell, solute, molar_mass: float, density: float
):
    return count_for_density(density, free_volume(cell, solute), molar_mass)


def charge_of(system: System | None) -> int:
    if system is None or system.n_atoms == 0:
        return 0
    return int(system.charges.sum())


def concentration_of(n_pairs: int, n_water: int) -> float:
    if n_water <= 0:
        return 0.0
    return n_pairs * WATER_PER_LITRE / n_water


def distance(a, b) -> float:
    return float(math.dist(a, b))
