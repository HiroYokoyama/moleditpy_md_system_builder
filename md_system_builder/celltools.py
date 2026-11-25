"""Operations on a periodic system: centre, wrap, unwrap, supercell, scale and
orthorhombic conversion.  All functions take a :class:`System` with a cell and
return a new :class:`System`; the input is never modified."""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Sequence

import numpy as np

from .cellmeta import PeriodicCell
from .system import System, concatenate


def _need_cell(system: System) -> PeriodicCell:
    if system.cell is None:
        raise ValueError(
            "The structure has no periodic cell. Set one on the Box tab first."
        )
    return system.cell


def centre(system: System, atoms: Sequence[int] | None = None) -> System:
    """Translate everything so the geometric centre of ``atoms`` (default: all) is
    at the cell centre."""
    cell = _need_cell(system)
    out = system.copy()
    idx = (
        np.arange(system.n_atoms)
        if atoms is None or len(atoms) == 0
        else np.asarray(atoms)
    )
    out.xyz = out.xyz + (cell.centre - out.xyz[idx].mean(axis=0))
    return out


def wrap_atoms(system: System) -> System:
    """Every atom into the cell, independently (molecules may be cut at the faces)."""
    cell = _need_cell(system)
    out = system.copy()
    frac = cell.to_fractional(out.xyz)
    mask = np.array(cell.pbc, dtype=float)
    frac -= np.floor(frac) * mask
    out.xyz = cell.to_cartesian(frac)
    return out


def wrap_molecules(system: System) -> System:
    """Each molecule shifted by whole cell vectors so its centre is in the cell;
    molecules stay intact (they are unwrapped first)."""
    cell = _need_cell(system)
    out = unwrap(system)
    mask = np.array(cell.pbc, dtype=float)
    for atoms in out.molecules():
        frac_c = cell.to_fractional(out.xyz[atoms].mean(axis=0))
        shift = np.floor(frac_c) * mask
        out.xyz[atoms] -= shift @ cell.lattice
    return out


def unwrap(system: System) -> System:
    """Make bonded molecules whole: walk the bond graph and put every neighbour at
    the minimum-image position relative to the atom it was reached from."""
    cell = _need_cell(system)
    out = system.copy()
    n = out.n_atoms
    neighbours = [[] for _ in range(n)]
    for i, j, _ in out.bonds:
        neighbours[int(i)].append(int(j))
        neighbours[int(j)].append(int(i))
    seen = np.zeros(n, dtype=bool)
    for start in range(n):
        if seen[start]:
            continue
        seen[start] = True
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b in neighbours[a]:
                if seen[b]:
                    continue
                delta = cell.minimum_image((out.xyz[b] - out.xyz[a]).reshape(1, 3))[0]
                out.xyz[b] = out.xyz[a] + delta
                seen[b] = True
                queue.append(b)
    return out


def supercell(system: System, repeats: Sequence[int]) -> System:
    """n_a x n_b x n_c copies; copies of the same molecule stay together in the atom
    order (all copies of molecule 1, then molecule 2, ...) so identical molecules
    remain consecutive and can be counted."""
    cell = _need_cell(system)
    reps = [int(r) for r in repeats]
    if len(reps) != 3 or min(reps) < 1:
        raise ValueError("Repeats must be three positive integers.")
    if np.prod(reps) > 1000:
        raise ValueError("At most 1000 copies.")
    shifts = [
        np.array(t, dtype=float) @ cell.lattice
        for t in itertools.product(range(reps[0]), range(reps[1]), range(reps[2]))
    ]
    pieces = []
    for atoms in system.molecules():
        mol = system.subset(atoms)
        for shift in shifts:
            copy = mol.copy()
            copy.xyz = copy.xyz + shift
            copy.cell = None
            pieces.append(copy)
    new_cell = PeriodicCell(
        cell.lattice * np.array(reps, dtype=float)[:, None], cell.origin, cell.pbc
    )
    out = concatenate(pieces, cell=new_cell)
    return out


def scale(
    system: System,
    factors: Sequence[float],
    coordinates: str = "molecules",
) -> System:
    """Scale the cell vectors a, b, c by ``factors``.

    ``coordinates``: ``"none"`` keeps Cartesian positions, ``"atoms"`` scales every
    atom's fractional position with the cell, ``"molecules"`` moves each molecule
    rigidly with its centre (bond lengths unchanged).
    """
    cell = _need_cell(system)
    f = np.asarray(factors, dtype=float).reshape(3)
    if np.any(f <= 0):
        raise ValueError("Scale factors must be positive.")
    new_cell = PeriodicCell(cell.lattice * f[:, None], cell.origin, cell.pbc)
    out = system.copy()
    out.cell = new_cell
    if coordinates == "none":
        return out
    if coordinates == "atoms":
        out.xyz = new_cell.to_cartesian(cell.to_fractional(system.xyz))
        return out
    if coordinates != "molecules":
        raise ValueError(f"Unknown coordinate mode {coordinates!r}.")
    for atoms in out.molecules():
        centre_old = out.xyz[atoms].mean(axis=0)
        centre_new = new_cell.to_cartesian(cell.to_fractional(centre_old))
        out.xyz[atoms] += centre_new - centre_old
    return out


def orthorhombic_transform(cell: PeriodicCell, max_index: int = 2, tol: float = 1e-3):
    """Smallest integer matrix M (rows: new vectors in terms of a, b, c) whose
    lattice M @ L is orthogonal, or None.  Searches entries in [-max_index, max_index]."""
    lattice = cell.lattice
    candidates = []
    rng = range(-max_index, max_index + 1)
    for row in itertools.product(rng, repeat=3):
        if row == (0, 0, 0):
            continue
        vec = np.array(row, dtype=float) @ lattice
        candidates.append((np.linalg.norm(vec), row, vec))
    candidates.sort(key=lambda t: (t[0], sum(abs(v) for v in t[1])))
    best = None
    short = candidates[:60]
    for (n1, r1, v1), (n2, r2, v2), (n3, r3, v3) in itertools.combinations(short, 3):
        if abs(np.dot(v1, v2)) > tol * n1 * n2:
            continue
        if abs(np.dot(v1, v3)) > tol * n1 * n3 or abs(np.dot(v2, v3)) > tol * n2 * n3:
            continue
        m = np.array([r1, r2, r3], dtype=int)
        det = round(np.linalg.det(m))
        if det == 0:
            continue
        key = (abs(det), n1 * n2 * n3)
        if best is None or key < best[0]:
            best = (key, m if det > 0 else -m)
    return None if best is None else best[1]


def to_orthorhombic(system: System, max_index: int = 2) -> System:
    """Equivalent orthorhombic cell (possibly larger), rotated so a, b, c lie on x, y, z.

    Raises ValueError when no integer combination of the cell vectors up to
    ``max_index`` is orthogonal (e.g. a general triclinic cell).
    """
    cell = _need_cell(system)
    if cell.is_orthorhombic():
        return system.copy()
    m = orthorhombic_transform(cell, max_index)
    if m is None:
        raise ValueError(
            "No orthorhombic cell found: no combination of the cell vectors with "
            f"coefficients up to {max_index} is mutually perpendicular."
        )
    new_lattice = m @ cell.lattice
    n_copies = round(abs(np.linalg.det(m)))
    whole = wrap_molecules(system)
    # Enough translates of the old cell to cover the new one.
    corners = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float) @ m
    lo = np.floor(corners.min(axis=0)).astype(int) - 1
    hi = np.ceil(corners.max(axis=0)).astype(int) + 1
    new_cell_raw = PeriodicCell(new_lattice, cell.origin)
    pieces = []
    for atoms in whole.molecules():
        mol = whole.subset(atoms)
        centre_c = mol.xyz.mean(axis=0)
        for t in itertools.product(*(range(a, b) for a, b in zip(lo, hi))):
            shift = np.array(t, dtype=float) @ cell.lattice
            frac = new_cell_raw.to_fractional(centre_c + shift)
            if np.all(frac >= -1e-9) and np.all(frac < 1 - 1e-9):
                copy = mol.copy()
                copy.xyz = copy.xyz + shift
                copy.cell = None
                pieces.append(copy)
    expected = whole.n_molecules * n_copies
    if len(pieces) != expected:
        raise ValueError(
            f"Orthorhombic conversion found {len(pieces)} molecules, expected {expected}."
        )
    # Rotate so the new vectors are along x, y, z.
    axes = new_lattice / np.linalg.norm(new_lattice, axis=1)[:, None]
    rot = axes  # rows are the new unit axes; xyz @ rot.T gives coordinates on them
    out = concatenate(pieces)
    out.xyz = (out.xyz - cell.origin) @ rot.T
    lengths = np.linalg.norm(new_lattice, axis=1)
    out.cell = PeriodicCell(np.diag(lengths), np.zeros(3), cell.pbc)
    return out
