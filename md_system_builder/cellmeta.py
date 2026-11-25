"""The suite-wide periodic cell record: the ``moleditpy_cell`` molecule property.

``{"lattice": [[ax, ay, az], [bx, by, bz], [cx, cy, cz]], "origin": [x, y, z],
"pbc": [true, true, true]}`` in Angstrom, rows are the cell vectors.  Builders set
it on the molecule they create; the LAMMPS / GROMACS / OpenMM generators read it.

Pure numpy + json; RDKit molecules are only duck-typed (``HasProp`` / ``GetProp`` /
``SetProp`` / ``ClearProp``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .cell_model import cell_vectors, lattice_parameters

PROP = "moleditpy_cell"


@dataclass
class PeriodicCell:
    lattice: np.ndarray
    origin: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pbc: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self):
        self.lattice = np.asarray(self.lattice, dtype=float).reshape(3, 3)
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.pbc = tuple(bool(v) for v in self.pbc)
        if not np.all(np.isfinite(self.lattice)) or not np.all(
            np.isfinite(self.origin)
        ):
            raise ValueError("The cell contains non-finite numbers.")
        if abs(self.volume) < 1e-6:
            raise ValueError("The cell vectors are linearly dependent (zero volume).")

    # -- derived quantities ----------------------------------------------

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def parameters(self):
        """((a, b, c), (alpha, beta, gamma))."""
        return lattice_parameters(self.lattice)

    @property
    def centre(self) -> np.ndarray:
        return self.origin + 0.5 * self.lattice.sum(axis=0)

    @property
    def widths(self) -> np.ndarray:
        """Perpendicular widths of the cell along a, b and c (distance between faces)."""
        a, b, c = self.lattice
        vol = self.volume
        return np.array(
            [
                vol / np.linalg.norm(np.cross(b, c)),
                vol / np.linalg.norm(np.cross(c, a)),
                vol / np.linalg.norm(np.cross(a, b)),
            ]
        )

    def is_orthorhombic(self, tol: float = 1e-4) -> bool:
        off = self.lattice - np.diag(np.diag(self.lattice))
        return bool(np.all(np.abs(off) < tol))

    def to_fractional(self, xyz) -> np.ndarray:
        xyz = np.asarray(xyz, dtype=float)
        return np.linalg.solve(self.lattice.T, (xyz - self.origin).T).T

    def to_cartesian(self, frac) -> np.ndarray:
        frac = np.asarray(frac, dtype=float)
        return frac @ self.lattice + self.origin

    def minimum_image(self, delta) -> np.ndarray:
        """Shift Cartesian difference vectors to their nearest periodic image."""
        delta = np.asarray(delta, dtype=float)
        frac = np.linalg.solve(self.lattice.T, delta.T).T
        mask = np.array(self.pbc, dtype=float)
        frac -= np.round(frac) * mask
        return frac @ self.lattice

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "lattice": [[round(float(v), 8) for v in row] for row in self.lattice],
            "origin": [round(float(v), 8) for v in self.origin],
            "pbc": [bool(v) for v in self.pbc],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data) -> PeriodicCell:
        if not isinstance(data, dict) or "lattice" not in data:
            raise ValueError("A cell record needs a 'lattice'.")
        lattice = np.asarray(data["lattice"], dtype=float)
        if lattice.shape != (3, 3):
            raise ValueError("The cell lattice must be 3 x 3.")
        origin = data.get("origin") or [0.0, 0.0, 0.0]
        pbc = data.get("pbc") or [True, True, True]
        if len(origin) != 3 or len(pbc) != 3:
            raise ValueError("The cell origin and pbc must have three entries.")
        return cls(lattice, origin, tuple(bool(v) for v in pbc))

    @classmethod
    def from_json(cls, text: str) -> PeriodicCell:
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_parameters(
        cls, lengths: Sequence[float], angles: Sequence[float], origin=None
    ):
        return cls(
            cell_vectors(lengths, angles), np.zeros(3) if origin is None else origin
        )

    @classmethod
    def from_lengths(cls, lengths: Sequence[float], origin=None):
        return cls(
            np.diag([float(v) for v in lengths]),
            np.zeros(3) if origin is None else origin,
        )

    def copy(self) -> PeriodicCell:
        return PeriodicCell(self.lattice.copy(), self.origin.copy(), self.pbc)

    def describe(self) -> str:
        (a, b, c), (al, be, ga) = self.parameters
        return (
            f"a={a:.3f} b={b:.3f} c={c:.3f} A, alpha={al:.2f} beta={be:.2f} "
            f"gamma={ga:.2f} deg, V={self.volume:.1f} A^3"
        )


def bounding_box_cell(xyz, padding: float, cubic: bool = False) -> PeriodicCell:
    """Orthorhombic cell around the points with ``padding`` on every side."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if len(xyz) == 0:
        raise ValueError("There are no atoms to put in a box.")
    if padding < 0 or not math.isfinite(padding):
        raise ValueError("The padding must be a non-negative number.")
    lo = xyz.min(axis=0) - padding
    hi = xyz.max(axis=0) + padding
    size = np.maximum(hi - lo, 1.0)
    if cubic:
        centre = 0.5 * (lo + hi)
        size = np.full(3, size.max())
        lo = centre - size / 2
    return PeriodicCell(np.diag(size), lo)


def read_cell(mol) -> PeriodicCell | None:
    """The molecule's cell, or None when it has none (or an unreadable one)."""
    if mol is None:
        return None
    try:
        if not mol.HasProp(PROP):
            return None
        return PeriodicCell.from_json(mol.GetProp(PROP))
    except Exception:  # noqa: BLE001 - a malformed record is treated as no cell
        return None


def write_cell(mol, cell: PeriodicCell | None) -> None:
    if mol is None:
        return
    if cell is None:
        if mol.HasProp(PROP):
            mol.ClearProp(PROP)
        return
    mol.SetProp(PROP, cell.to_json())
