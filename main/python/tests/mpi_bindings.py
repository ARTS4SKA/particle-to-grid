"""Multi-rank tests for the particle_to_grid python module.

Run under MPI, e.g. `mpiexec -n 4 python mpi_bindings.py`. Exits non-zero on the first failure.
This covers what the serial pytest run cannot: the cross-rank MPI exchange inside p2g and g2p.

Uses only numpy, not mpi4py: the module initializes MPI itself. mpi4py is imported if available
purely to cross-check the rank/size the module reports.
"""

import sys

import numpy as np

import particle_to_grid as p2g

RANK = p2g.mpi_rank()
SIZE = p2g.mpi_size()
GRID_DIM = 8
LMIN, LMAX = 0.0, 1.0

failures = []
skipped = []

try:
    from mpi4py import MPI

    HAVE_MPI4PY = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_MPI4PY = False


def check(condition, message):
    if not condition:
        failures.append(message)


def report(name):
    if RANK == 0:
        print(f"  {name}: {'ok' if not failures else 'FAILED'}", flush=True)


def skip(name, reason):
    skipped.append(name)
    if RANK == 0:
        print(f"  {name}: SKIPPED ({reason})", flush=True)


def particles(n, seed):
    """Identical particle set on every rank, split into this rank's contiguous share."""
    rng = np.random.default_rng(seed)
    x, y, z = (rng.uniform(LMIN, LMAX, n) for _ in range(3))
    mass = rng.uniform(0.5, 1.5, n)
    temp = rng.normal(100.0, 10.0, n)

    # Each rank takes a slice; together the ranks cover every particle exactly once.
    lo = (n * RANK) // SIZE
    hi = (n * (RANK + 1)) // SIZE
    return x[lo:hi], y[lo:hi], z[lo:hi], mass[lo:hi], temp[lo:hi], (x, y, z, mass, temp)


def global_sum(value):
    """Sum a scalar across ranks. Needs mpi4py; callers must handle None."""
    if not HAVE_MPI4PY:
        return None
    return MPI.COMM_WORLD.allreduce(value, op=MPI.SUM)


def test_mpi_state_matches_mpi4py():
    """The module shares MPI_COMM_WORLD with mpi4py rather than starting its own."""
    if not HAVE_MPI4PY:
        skip("mpi state matches mpi4py", "mpi4py not installed")
        return
    check(MPI.COMM_WORLD.Get_rank() == RANK, "module rank disagrees with mpi4py")
    check(MPI.COMM_WORLD.Get_size() == SIZE, "module size disagrees with mpi4py")
    report("mpi state matches mpi4py")


def test_density_conserves_mass():
    """Mass deposited across ranks must add up to the total particle mass."""
    x, y, z, mass, _, (_, _, _, all_mass, _) = particles(20000, seed=1)

    grid = p2g.CartesianGrid(grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX)
    grid.set_reduction(0, p2g.Reduction.Density)
    grid.p2g(p2g.hilbert_keys(x, y, z, lmin=LMIN, lmax=LMAX), [mass])

    local_mass = float((grid.field(0) * grid.cell_volume).sum())
    total = global_sum(local_mass)
    if total is None:
        skip("density conserves mass across ranks", "needs mpi4py to sum across ranks")
        return
    check(
        abs(total - all_mass.sum()) / all_mass.sum() < 1e-12,
        f"mass not conserved: grid={total} particles={all_mass.sum()}",
    )
    report("density conserves mass across ranks")


def test_residual_round_trip_across_ranks():
    """Cells are owned by other ranks, so this exercises both exchange directions."""
    x, y, z, mass, temp, _ = particles(20000, seed=2)
    keys = p2g.hilbert_keys(x, y, z, lmin=LMIN, lmax=LMAX)

    grid = p2g.CartesianGrid(grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX)
    grid.ensure_num_fields(2)

    residuals = grid.p2g(keys, [mass, temp], residuals=True)
    out = grid.g2p(keys, residuals=residuals)

    check(np.allclose(out[0], mass, rtol=1e-14), "mass not recovered through the residual round trip")
    check(np.allclose(out[1], temp, rtol=1e-14), "temp not recovered through the residual round trip")

    # Without residuals the result is the cell mean, which must differ for shared cells.
    lossy = grid.g2p(keys, num_fields=2)
    check(not np.allclose(lossy[0], mass), "cell means unexpectedly equal the inputs")
    report("residual round trip exact across ranks")


def test_slab_ownership_is_disjoint_and_complete():
    """Each rank owns gridDim/numRanks planes, and the local slabs tile the full grid."""
    grid = p2g.CartesianGrid(grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX)
    expected = GRID_DIM * GRID_DIM * (GRID_DIM // SIZE)
    check(grid.local_size == expected, f"local_size {grid.local_size} != {expected}")

    total = global_sum(grid.local_size)
    if total is None:
        skip("slab coverage", "needs mpi4py to sum across ranks")
    else:
        check(total == GRID_DIM**3, f"slabs cover {total} cells, expected {GRID_DIM**3}")

    planes = [k for k in range(GRID_DIM) if p2g.slab_rank(k, GRID_DIM, SIZE) == RANK]
    check(len(planes) == GRID_DIM // SIZE, "unexpected number of owned planes")
    report("slab ownership disjoint and complete")


def main():
    if GRID_DIM % SIZE != 0:
        if RANK == 0:
            print(f"grid_dim {GRID_DIM} is not divisible by {SIZE} ranks", file=sys.stderr)
        return 1

    if RANK == 0:
        print(f"particle_to_grid python bindings, {SIZE} rank(s):", flush=True)

    for test in (
        test_mpi_state_matches_mpi4py,
        test_density_conserves_mass,
        test_residual_round_trip_across_ranks,
        test_slab_ownership_is_disjoint_and_complete,
    ):
        before = len(failures)
        test()
        if len(failures) > before:
            break

    if failures:
        print(f"rank {RANK} FAILED:", *failures, sep="\n  ", file=sys.stderr)
        return 1
    if RANK == 0 and skipped:
        print(f"  ({len(skipped)} check(s) skipped: install mpi4py to run them)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
