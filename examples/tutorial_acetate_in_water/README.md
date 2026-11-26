# Files of the acetate-in-water tutorial

The input and output of the tutorial
[acetate in water, from drawing to a GROMACS run](../../docs/tutorial_acetate_in_water.md),
so each step can be checked or repeated without redoing the earlier ones.

| Path | What |
|---|---|
| `acetate.mol` | the solute (step 1) |
| `system.mdff.json` | the force field of the solvated system, saved by Force Field Builder (step 3) |
| `input/` | everything the GROMACS Input Generator wrote (step 4): `conf.gro`, `topol.top`, the `.itp` files, `em/nvt/npt/md.mdp`, `index.ndx`, `run.sh`, `run.bat` |
| `results/run.log` | output of `bash run.sh` with GROMACS 2025.3 (step 5) |
| `results/md.log`, `results/md.edr` | the production run's log and energies (`gmx energy -f md.edr`) |
| `results/md.gro`, `results/md.xtc` | final structure and the 1 ns trajectory, one frame every 10 ps |
| `results/md_traj_50ps.gro` | the trajectory every 50 ps (21 frames) as GRO, to open directly in MoleditPy (step 6) |

The full-resolution GRO trajectory used for the analyses in the tutorial is made from
`md.xtc` with `gmx trjconv` (see step 6); it is not stored here because of its size.
Paths in the logs were shortened.
