"""Packmol-like random packing of rigid molecules into a periodic cell (numpy only).

The packing core is shared by the Pack and Solvate tabs.  Each copy of a component
gets a uniform random orientation and a random position inside its region; it is
accepted when no atom comes closer than ``tolerance`` to an atom already placed,
with periodic images taken into account.  Overlaps are found with a cell list over
fractional coordinates, so the cost per trial is constant however full the box is.

There is no optimisation stage: a box that is too full for random insertion fails
with a message saying how far it got, instead of producing overlapping atoms.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .cellmeta import PeriodicCell
from .elements import atomic_mass
from .system import System, concatenate

AVOGADRO_A3 = 0.602214076  # (g/mol) / (g/cm3) <-> molecules per A^3 factor

REGION_BOX = "box"
REGION_INSIDE_SPHERE = "inside_sphere"
REGION_OUTSIDE_SPHERE = "outside_sphere"
REGION_INSIDE_SLAB = "inside_slab"
REGION_OUTSIDE_SLAB = "outside_slab"
REGIONS = (
    REGION_BOX,
    REGION_INSIDE_SPHERE,
    REGION_OUTSIDE_SPHERE,
    REGION_INSIDE_SLAB,
    REGION_OUTSIDE_SLAB,
)


class PackingError(RuntimeError):
    """The requested packing could not be completed."""


class Cancelled(RuntimeError):
    """The user cancelled the job."""


@dataclass
class Region:
    """Where the atoms of a component may go, in Cartesian Angstrom.

    Sphere: ``centre`` + ``radius``.  Slab: positions whose coordinate along ``axis``
    (0, 1, 2 = x, y, z) lies between ``low`` and ``high``.  ``None`` centre means the
    cell centre.
    """

    kind: str = REGION_BOX
    centre: Sequence[float] | None = None
    radius: float = 10.0
    axis: int = 2
    low: float = 0.0
    high: float = 10.0

    def __post_init__(self):
        if self.kind not in REGIONS:
            raise ValueError(f"Unknown region {self.kind!r}.")
        if (
            self.kind in (REGION_INSIDE_SPHERE, REGION_OUTSIDE_SPHERE)
            and self.radius <= 0
        ):
            raise ValueError("The sphere radius must be positive.")
        if (
            self.kind in (REGION_INSIDE_SLAB, REGION_OUTSIDE_SLAB)
            and self.high <= self.low
        ):
            raise ValueError("The slab needs high > low.")

    def satisfied(self, xyz: np.ndarray, cell: PeriodicCell) -> bool:
        if self.kind == REGION_BOX:
            return True
        if self.kind in (REGION_INSIDE_SPHERE, REGION_OUTSIDE_SPHERE):
            centre = (
                cell.centre if self.centre is None else np.asarray(self.centre, float)
            )
            d = np.sqrt(((xyz - centre) ** 2).sum(axis=1))
            if self.kind == REGION_INSIDE_SPHERE:
                return bool(np.all(d <= self.radius))
            return bool(np.all(d >= self.radius))
        coord = xyz[:, int(self.axis)]
        inside = (coord >= self.low) & (coord <= self.high)
        if self.kind == REGION_INSIDE_SLAB:
            return bool(np.all(inside))
        return bool(not np.any(inside))

    def sample_centre(self, rng, cell: PeriodicCell) -> np.ndarray:
        """A random point that is likely to satisfy the region."""
        if self.kind == REGION_INSIDE_SPHERE:
            centre = (
                cell.centre if self.centre is None else np.asarray(self.centre, float)
            )
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            return centre + direction * self.radius * rng.random() ** (1 / 3)
        point = cell.to_cartesian(rng.random(3))
        if self.kind == REGION_INSIDE_SLAB:
            point[int(self.axis)] = rng.uniform(self.low, self.high)
        return point

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "centre": None if self.centre is None else [float(v) for v in self.centre],
            "radius": float(self.radius),
            "axis": int(self.axis),
            "low": float(self.low),
            "high": float(self.high),
        }

    @classmethod
    def from_dict(cls, data) -> Region:
        data = dict(data or {})
        return cls(
            kind=data.get("kind", REGION_BOX),
            centre=data.get("centre"),
            radius=float(data.get("radius", 10.0)),
            axis=int(data.get("axis", 2)),
            low=float(data.get("low", 0.0)),
            high=float(data.get("high", 10.0)),
        )


@dataclass
class Component:
    """One molecule species to pack: a template (a single molecule) and a count."""

    template: System
    count: int = 1
    name: str = "MOL"
    region: Region = field(default_factory=Region)

    def __post_init__(self):
        if self.template.n_atoms == 0:
            raise ValueError(f"Component {self.name} has no atoms.")
        if self.count < 0:
            raise ValueError("Counts cannot be negative.")
        centred = self.template.copy()
        centred.xyz = centred.xyz - centred.xyz.mean(axis=0)
        centred.cell = None
        self.template = centred

    @property
    def mass(self) -> float:
        return self.template.total_mass()


def count_for_density(density_g_cm3: float, volume_a3: float, molar_mass: float) -> int:
    """Number of molecules of ``molar_mass`` that give ``density`` in ``volume``."""
    if density_g_cm3 <= 0 or volume_a3 <= 0 or molar_mass <= 0:
        return 0
    return round(density_g_cm3 * volume_a3 * AVOGADRO_A3 / molar_mass)


def mixture_counts(density_g_cm3, volume_a3, components, fractions) -> list[int]:
    """Counts that give ``density`` with molecule-number ``fractions`` (normalised)."""
    fractions = np.asarray(fractions, dtype=float)
    if fractions.sum() <= 0:
        raise ValueError("Give at least one positive mole fraction.")
    fractions = fractions / fractions.sum()
    mean_mass = float(sum(f * c.mass for f, c in zip(fractions, components)))
    total = density_g_cm3 * volume_a3 * AVOGADRO_A3 / mean_mass
    return [round(total * f) for f in fractions]


def random_rotation(rng) -> np.ndarray:
    """Uniformly distributed rotation matrix (from a random unit quaternion)."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class CellGrid:
    """Cell list over fractional coordinates with periodic wrap-around.

    Bins are at least ``cutoff`` wide in every direction (measured perpendicular to
    the faces), so neighbours within ``cutoff`` are always in the 27 surrounding bins.
    """

    def __init__(self, cell: PeriodicCell, cutoff: float):
        self.cell = cell
        self.cutoff = float(cutoff)
        self.inv = np.linalg.inv(cell.lattice)
        widths = cell.widths
        self.nbins = np.maximum(
            1, np.floor(widths / max(self.cutoff, 1e-6)).astype(int)
        )
        self.bins = {}
        self.xyz = np.zeros((0, 3))
        self._chunks: list[np.ndarray] = []
        self._count = 0

    def _frac(self, xyz):
        return (np.asarray(xyz, dtype=float) - self.cell.origin) @ self.inv

    def _keys(self, frac):
        wrapped = frac - np.floor(frac)
        idx = np.floor(wrapped * self.nbins).astype(int)
        return np.minimum(idx, self.nbins - 1)

    def _materialise(self):
        if self._chunks:
            self.xyz = np.vstack([self.xyz] + self._chunks)
            self._chunks = []

    def add(self, xyz) -> None:
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        if len(xyz) == 0:
            return
        keys = self._keys(self._frac(xyz))
        start = self._count
        for k, key in enumerate(map(tuple, keys)):
            self.bins.setdefault(key, []).append(start + k)
        self._chunks.append(xyz)
        self._count += len(xyz)

    def neighbours(self, xyz) -> np.ndarray:
        """Indices of stored points that may lie within ``cutoff`` of any query point."""
        self._materialise()
        keys = self._keys(self._frac(np.asarray(xyz, dtype=float).reshape(-1, 3)))
        found = set()
        nb = self.nbins
        offsets = [(-1, 0, 1) if n >= 3 else tuple(range(n)) for n in nb]
        for key in {tuple(k) for k in keys}:
            for dx in offsets[0]:
                for dy in offsets[1]:
                    for dz in offsets[2]:
                        if nb[0] >= 3:
                            bx = (key[0] + dx) % nb[0]
                        else:
                            bx = dx
                        if nb[1] >= 3:
                            by = (key[1] + dy) % nb[1]
                        else:
                            by = dy
                        if nb[2] >= 3:
                            bz = (key[2] + dz) % nb[2]
                        else:
                            bz = dz
                        hit = self.bins.get((bx, by, bz))
                        if hit:
                            found.update(hit)
        return np.fromiter(found, dtype=int, count=len(found))

    def clashes(self, xyz, cutoff: float | None = None) -> bool:
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        idx = self.neighbours(xyz)
        if len(idx) == 0:
            return False
        limit = self.cutoff if cutoff is None else float(cutoff)
        delta = self.xyz[idx][None, :, :] - xyz[:, None, :]
        delta = self.cell.minimum_image(delta.reshape(-1, 3))
        return bool(np.any((delta * delta).sum(axis=1) < limit * limit))

    def min_distance(self, xyz) -> np.ndarray:
        """Per query point, the distance to the nearest stored point (inf beyond cutoff bins)."""
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        out = np.full(len(xyz), np.inf)
        for k, point in enumerate(xyz):
            idx = self.neighbours(point)
            if len(idx):
                delta = self.cell.minimum_image(self.xyz[idx] - point)
                out[k] = math.sqrt(float((delta * delta).sum(axis=1).min()))
        return out

    def __len__(self):
        return self._count


@dataclass
class PackResult:
    system: System
    placed: list[int]
    attempts: int


def pack(
    cell: PeriodicCell,
    components: Sequence[Component],
    tolerance: float = 2.0,
    seed: int | None = None,
    fixed: System | None = None,
    centre_fixed: bool = True,
    max_attempts: int = 2000,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    allow_partial: bool = False,
) -> PackResult:
    """Place ``count`` copies of each component in ``cell``.

    ``fixed`` (optional) is kept rigid; with ``centre_fixed`` it is translated so its
    geometric centre sits at the cell centre.  ``max_attempts`` is the number of
    random trials per molecule before giving up.  Raises :class:`PackingError` with
    a description of how far it got, unless ``allow_partial``.
    """
    if tolerance <= 0:
        raise ValueError("The tolerance must be positive.")
    rng = np.random.default_rng(seed)
    grid = CellGrid(cell, tolerance)
    parts: list[System] = []

    if fixed is not None and fixed.n_atoms:
        solute = fixed.copy()
        if centre_fixed:
            solute.xyz = solute.xyz - solute.xyz.mean(axis=0) + cell.centre
        solute.cell = None
        grid.add(solute.xyz)
        parts.append(solute)

    total = int(sum(c.count for c in components))
    done = 0
    attempts_total = 0
    placed_counts = []
    for comp in components:
        placed = 0
        tpl = comp.template
        copies = []
        for _ in range(comp.count):
            if cancelled is not None and cancelled():
                raise Cancelled("Cancelled.")
            ok = False
            for _attempt in range(max_attempts):
                attempts_total += 1
                rot = random_rotation(rng)
                centre = comp.region.sample_centre(rng, cell)
                xyz = tpl.xyz @ rot.T + centre
                if not comp.region.satisfied(xyz, cell):
                    continue
                if grid.clashes(xyz):
                    continue
                grid.add(xyz)
                copy = tpl.copy()
                copy.xyz = xyz
                copies.append(copy)
                ok = True
                break
            if not ok:
                if allow_partial:
                    break
                raise PackingError(
                    f"Placed {placed} of {comp.count} copies of {comp.name} "
                    f"({done} of {total} molecules in all) before {max_attempts} "
                    "random trials in a row failed. The box is too full for this "
                    f"tolerance ({tolerance:.2f} A): enlarge the cell, lower the "
                    "count or density, reduce the tolerance or raise the retry limit."
                )
            placed += 1
            done += 1
            if progress is not None and (done % 25 == 0 or done == total):
                progress(done, total)
        placed_counts.append(placed)
        if copies:
            parts.append(_stack_copies(copies, comp.name))

    system = concatenate(parts, cell=cell.copy())
    return PackResult(system, placed_counts, attempts_total)


def _stack_copies(copies: list[System], name: str) -> System:
    """Copies of one template as one System; residue numbers 1..n, one per copy."""
    tpl = copies[0]
    n = tpl.n_atoms
    k = len(copies)
    single_residue = len(set(tpl.resnums.tolist())) <= 1
    resname = name[:4] if name else tpl.resnames[0]
    system = concatenate(copies)
    if single_residue:
        system.resnames = [resname] * (n * k)
        system.resnums = np.repeat(np.arange(1, k + 1), n)
    return system


def molar_mass(elements) -> float:
    return float(sum(atomic_mass(e) for e in elements))


def free_volume(
    cell: PeriodicCell,
    solute: System | None,
    probe: float = 1.6,
    samples: int = 20000,
    seed: int = 0,
) -> float:
    """Cell volume not occupied by the solute, estimated by random sampling.

    A sample point counts as occupied when it is within ``probe`` Angstrom of a
    solute atom: about a heavy-atom van der Waals radius, so the result is the
    volume a solvent could fill.
    """
    if solute is None or solute.n_atoms == 0:
        return cell.volume
    grid = CellGrid(cell, probe)
    grid.add(solute.xyz)
    rng = np.random.default_rng(seed)
    points = cell.to_cartesian(rng.random((samples, 3)))
    occupied = 0
    for point in points:
        if grid.clashes(point):
            occupied += 1
    return cell.volume * (1.0 - occupied / samples)
