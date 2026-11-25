"""Generate the pre-equilibrated TIP3P water box shipped in ``md_system_builder/water_box.py``.

Rigid TIP3P water, 216 molecules in a cubic box at 0.997 g/cm3 (298 K), equilibrated
by Metropolis Monte Carlo (single-molecule translations and rotations).  LJ between
oxygens, Coulomb between all sites with a molecule-based cutoff and a reaction field,
minimum image.  The result is a liquid-like configuration (first O-O peak near 2.8
Angstrom) that is tiled into a cell and then relaxed by the user's MD engine; it is
not meant to reproduce TIP3P thermodynamics exactly.

Not part of the plugin.  Run it from the repository root:

    python tools/generate_water_box.py [--sweeps 3000] [--seed 7]

and it rewrites ``md_system_builder/water_box.py``.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

N_MOL = 216
DENSITY = 0.997  # g/cm3
MASS = 18.01528  # g/mol
R_OH = 0.9572
ANGLE = math.radians(104.52)
Q_O, Q_H = -0.834, 0.417
SIGMA, EPS = 3.15061, 0.1521  # Angstrom, kcal/mol
COULOMB = 332.0637  # kcal Angstrom / (mol e^2)
KT = 0.0019872041 * 298.15
CUTOFF = 9.0
EPS_RF = 78.5

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "md_system_builder",
    "water_box.py",
)


def template():
    half = ANGLE / 2.0
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [R_OH * math.sin(half), R_OH * math.cos(half), 0.0],
            [-R_OH * math.sin(half), R_OH * math.cos(half), 0.0],
        ]
    )


def random_rotation(rng):
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


def small_rotation(rng, max_angle):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-max_angle, max_angle)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


CHARGES = np.array([Q_O, Q_H, Q_H])
QQ = np.outer(CHARGES, CHARGES)  # 3x3
K_RF = (EPS_RF - 1) / ((2 * EPS_RF + 1) * CUTOFF**3)
C_RF = 1.0 / CUTOFF + K_RF * CUTOFF**2


def mol_energy(i, pos_i, sites, box):
    """Interaction energy of molecule i (sites pos_i, 3x3) with all others."""
    d_oo = sites[:, 0, :] - pos_i[0]
    shift = -box * np.round(d_oo / box)
    d_oo = d_oo + shift
    r_oo = np.sqrt((d_oo * d_oo).sum(axis=1))
    mask = r_oo < CUTOFF
    mask[i] = False
    if not mask.any():
        return 0.0
    r = r_oo[mask]
    sr6 = (SIGMA / r) ** 6
    e_lj = (4 * EPS * (sr6 * sr6 - sr6)).sum()
    other = sites[mask] + shift[mask][:, None, :]  # m x 3 x 3
    diff = other[:, None, :, :] - pos_i[None, :, None, :]  # m x 3(i) x 3(j) x 3
    rij = np.sqrt((diff * diff).sum(axis=-1))
    e_c = (COULOMB * QQ[None] * (1.0 / rij + K_RF * rij * rij - C_RF)).sum()
    return e_lj + e_c


def rdf_oo(sites, box, bins=60, rmax=6.0):
    o = sites[:, 0, :]
    d = o[:, None, :] - o[None, :, :]
    d -= box * np.round(d / box)
    r = np.sqrt((d * d).sum(-1))[np.triu_indices(len(o), 1)]
    hist, edges = np.histogram(r, bins=bins, range=(0, rmax))
    rho = len(o) / box**3
    shell = 4 / 3 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (shell * rho * len(o) / 2)
    return 0.5 * (edges[1:] + edges[:-1]), g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    box = (N_MOL * MASS / (DENSITY * 0.602214076)) ** (1 / 3)
    n_side = 6
    spacing = box / n_side
    tpl = template()
    sites = np.empty((N_MOL, 3, 3))
    k = 0
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                centre = (np.array([ix, iy, iz]) + 0.5) * spacing
                sites[k] = tpl @ random_rotation(rng).T + centre
                k += 1

    max_disp, max_rot = 0.25, math.radians(20)
    accepted = 0
    started = time.time()
    for sweep in range(args.sweeps):
        for _ in range(N_MOL):
            i = int(rng.integers(N_MOL))
            old = sites[i]
            e_old = mol_energy(i, old, sites, box)
            rot = small_rotation(rng, max_rot)
            new = (old - old[0]) @ rot.T + old[0] + rng.uniform(-max_disp, max_disp, 3)
            e_new = mol_energy(i, new, sites, box)
            if e_new <= e_old or rng.random() < math.exp(-(e_new - e_old) / KT):
                sites[i] = new
                accepted += 1
        if (sweep + 1) % 250 == 0:
            ratio = accepted / ((sweep + 1) * N_MOL)
            print(
                f"sweep {sweep + 1}: acceptance {ratio:.2f}, {time.time() - started:.0f} s",
                flush=True,
            )

    # Wrap each molecule by its oxygen into the box.
    sites -= box * np.floor(sites[:, :1, :] / box)
    r, g = rdf_oo(sites, box)
    peak = r[int(np.argmax(g))]
    print(f"box {box:.4f} A, first g_OO peak at {peak:.2f} A (height {g.max():.2f})")

    rows = sites.reshape(-1, 3)
    lines = [
        '"""Pre-equilibrated TIP3P water box (216 molecules, 298 K, 0.997 g/cm3).',
        "",
        "Generated by ``tools/generate_water_box.py`` in this repository (rigid TIP3P,",
        f"Metropolis Monte Carlo, {args.sweeps} sweeps, seed {args.seed}).  Written by this",
        "project; no third-party coordinates.  Atom order O, H, H per molecule, Angstrom,",
        "every oxygen inside [0, BOX_LENGTH).",
        '"""',
        "",
        f"BOX_LENGTH = {box:.4f}",
        f"N_MOLECULES = {N_MOL}",
        "",
        "# x y z per atom",
        "COORDINATES = (",
    ]
    for x, y, z in rows:
        lines.append(f'    "{x:.3f} {y:.3f} {z:.3f}",')
    lines.append(")")
    with open(OUT, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
