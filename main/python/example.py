"""Minimal example of the particle_to_grid python module.

    PYTHONPATH=build/main/python python main/python/example.py
    PYTHONPATH=build/main/python mpiexec -n 4 python main/python/example.py
"""

import numpy as np

import particle_to_grid as p2g

GRID_DIM = 16
LMIN, LMAX = -0.5, 0.5

rank, size = p2g.mpi_rank(), p2g.mpi_size()

# Each rank contributes its own particles; the grid is split into z-slabs across ranks.
rng = np.random.default_rng(42 + rank)
n = 50_000
x, y, z = (rng.uniform(LMIN, LMAX, n) for _ in range(3))
mass = np.full(n, 1.0 / (n * size))
temperature = rng.normal(1e4, 1e3, n)

# --- one-liner: positions and masses straight to a density slab ---
rho = p2g.density(x, y, z, mass, grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX)
if rank == 0:
    print(f"density slab {rho.shape}, range [{rho.min():.4g}, {rho.max():.4g}]")

# --- full control: several fields, each with its own reduction ---
keys = p2g.hilbert_keys(x, y, z, lmin=LMIN, lmax=LMAX)

grid = p2g.CartesianGrid(grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX)
grid.ensure_num_fields(2)
grid.set_reduction(0, p2g.Reduction.Density)  # mass -> mass density
grid.set_reduction(1, p2g.Reduction.Mean)  # temperature -> cell average
grid.p2g(keys, [mass, temperature])

local_mass = (grid.field(0) * grid.cell_volume).sum()
if rank == 0:
    print(f"{grid!r}")
    print(f"rank 0 holds {local_mass:.6g} of the total mass")

# --- grid -> particles, with and without residuals ---
# Plain g2p is lossy: every particle in a cell gets that cell's value, so the
# per-particle spread collapses to the spread between cells.
cell_values = grid.g2p(keys, num_fields=2)
if rank == 0:
    print(f"temperature spread, particles: {temperature.std():.1f} -> cells: {cell_values[1].std():.1f}")

# Residuals make the round trip exact, at the cost of one array per field.
mean_grid = p2g.CartesianGrid(grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX)
mean_grid.ensure_num_fields(1)
residuals = mean_grid.p2g(keys, [temperature], residuals=True)
recovered, = mean_grid.g2p(keys, residuals=residuals)
if rank == 0:
    print(f"residual round trip exact: {np.array_equal(recovered, temperature)}")
