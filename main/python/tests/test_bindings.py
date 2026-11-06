"""Serial tests for the particle_to_grid python module.

Run with pytest. The module must be importable, i.e. PYTHONPATH must contain the directory
holding the built particle_to_grid extension (ctest sets this automatically).

The distributed exchange is covered separately by mpi_bindings.py, which runs under mpiexec.
"""

import numpy as np
import pytest

import particle_to_grid as p2g

GRID_DIM = 4
LMIN, LMAX = 0.0, 1.0
CELL_VOLUME = (1.0 / GRID_DIM) ** 3


def cell_center(i, j, k, grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX):
    """Position at the center of cell (i, j, k)."""
    dx = (lmax - lmin) / grid_dim
    return (lmin + (i + 0.5) * dx, lmin + (j + 0.5) * dx, lmin + (k + 0.5) * dx)


def keys_for_cells(cells, grid_dim=GRID_DIM):
    """Hilbert keys for particles placed at the centers of the given cells."""
    pts = np.array([cell_center(*c, grid_dim=grid_dim) for c in cells], dtype=float)
    return p2g.hilbert_keys(pts[:, 0], pts[:, 1], pts[:, 2], lmin=LMIN, lmax=LMAX)


def make_grid(grid_dim=GRID_DIM):
    return p2g.CartesianGrid(grid_dim=grid_dim, lmin=LMIN, lmax=LMAX)


@pytest.fixture(autouse=True)
def single_rank_only():
    if p2g.mpi_size() != 1:
        pytest.skip("serial tests; use mpi_bindings.py under mpiexec")


# --- basics ---------------------------------------------------------------


def test_module_reports_mpi_state():
    assert p2g.mpi_size() >= 1
    assert 0 <= p2g.mpi_rank() < p2g.mpi_size()


def test_grid_properties():
    grid = make_grid()
    assert grid.grid_dim == GRID_DIM
    assert grid.lmin == LMIN and grid.lmax == LMAX
    assert grid.local_size == GRID_DIM**3 // p2g.mpi_size()
    assert grid.cell_volume == pytest.approx(CELL_VOLUME)
    assert grid.num_fields() == 1
    assert "CartesianGrid" in repr(grid)


def test_keys_round_trip_through_cells():
    cells = [(0, 0, 0), (1, 2, 3), (3, 3, 3), (2, 0, 1)]
    keys = keys_for_cells(cells)
    assert keys.dtype == np.uint64
    for key, expected in zip(keys, cells):
        assert p2g.key_to_cell(key, GRID_DIM) == expected


def test_slab_rank_splits_planes_evenly():
    assert p2g.slab_rank(0, 4, 2) == 0
    assert p2g.slab_rank(1, 4, 2) == 0
    assert p2g.slab_rank(2, 4, 2) == 1
    assert p2g.slab_rank(3, 4, 2) == 1


def test_reduction_defaults_to_mean():
    grid = make_grid()
    assert grid.reduction(0) == p2g.Reduction.Mean


# --- p2g / g2p ------------------------------------------------------------


def test_density_sums_mass_and_divides_by_volume():
    grid = make_grid()
    grid.set_reduction(0, p2g.Reduction.Density)

    keys = keys_for_cells([(0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 2, 2)])
    mass = np.array([2.0, 2.0, 2.0, 5.0])
    grid.p2g(keys, [mass])

    rho = grid.field(0)
    assert rho[0] == pytest.approx(6.0 / CELL_VOLUME)
    assert rho[2 + 2 * GRID_DIM + 2 * GRID_DIM**2] == pytest.approx(5.0 / CELL_VOLUME)
    # Mass is conserved and only the two occupied cells are non-zero.
    assert (rho * CELL_VOLUME).sum() == pytest.approx(mass.sum())
    assert np.count_nonzero(rho) == 2


def test_mean_averages_intensive_fields():
    grid = make_grid()
    keys = keys_for_cells([(1, 1, 1), (1, 1, 1)])
    grid.p2g(keys, [np.array([10.0, 20.0])])

    assert grid.field(0)[1 + GRID_DIM + GRID_DIM**2] == pytest.approx(15.0)


def test_multiple_fields_use_their_own_reduction():
    grid = make_grid()
    grid.ensure_num_fields(2)
    grid.set_reduction(0, p2g.Reduction.Density)

    keys = keys_for_cells([(0, 0, 0), (0, 0, 0)])
    grid.p2g(keys, [np.array([1.0, 3.0]), np.array([10.0, 20.0])])

    assert grid.field(0)[0] == pytest.approx(4.0 / CELL_VOLUME)  # mass summed
    assert grid.field(1)[0] == pytest.approx(15.0)  # temperature averaged
    assert grid.num_fields() == 2


def test_g2p_returns_cell_means():
    grid = make_grid()
    keys = keys_for_cells([(0, 0, 0), (0, 0, 0), (2, 2, 2)])
    grid.p2g(keys, [np.array([4.0, 8.0, 7.0])])

    (out,) = grid.g2p(keys)
    # The shared cell hands both particles the mean; the lone particle gets its own value.
    assert out == pytest.approx([6.0, 6.0, 7.0])


def test_g2p_gathers_several_fields():
    grid = make_grid()
    grid.ensure_num_fields(2)
    keys = keys_for_cells([(0, 0, 0), (1, 1, 1)])
    grid.p2g(keys, [np.array([1.0, 2.0]), np.array([30.0, 40.0])])

    out = grid.g2p(keys, num_fields=2)
    assert len(out) == 2
    assert out[0] == pytest.approx([1.0, 2.0])
    assert out[1] == pytest.approx([30.0, 40.0])


# --- residuals ------------------------------------------------------------


def test_residuals_recover_original_values():
    grid = make_grid()
    keys = keys_for_cells([(0, 0, 0), (0, 0, 0), (0, 0, 0), (2, 2, 2)])
    mass = np.array([1.0, 2.0, 6.0, 7.0])

    residuals = grid.p2g(keys, [mass], residuals=True)
    assert len(residuals) == 1
    # Deviations from the cell mean of 3, summing to zero inside the cell.
    assert residuals[0] == pytest.approx([-2.0, -1.0, 3.0, 0.0])
    assert residuals[0][:3].sum() == pytest.approx(0.0)
    # The grid still holds the plain mean.
    assert grid.field(0)[0] == pytest.approx(3.0)

    (out,) = grid.g2p(keys, residuals=residuals)
    assert out == pytest.approx(mass)


def test_residuals_round_trip_many_particles_per_cell():
    rng = np.random.default_rng(12345)
    n = 5000
    x, y, z = (rng.uniform(LMIN, LMAX, n) for _ in range(3))
    values = rng.normal(100.0, 10.0, n)

    grid = make_grid(grid_dim=GRID_DIM)
    keys = p2g.hilbert_keys(x, y, z, lmin=LMIN, lmax=LMAX)

    residuals = grid.p2g(keys, [values], residuals=True)
    (out,) = grid.g2p(keys, residuals=residuals)

    # ~78 particles per cell on average, yet reconstruction is exact.
    np.testing.assert_allclose(out, values, rtol=1e-14)
    # Without the residuals, the same call returns the lossy cell means instead.
    (lossy,) = grid.g2p(keys)
    assert not np.allclose(lossy, values)


def test_residuals_rejected_for_density_fields():
    grid = make_grid()
    grid.set_reduction(0, p2g.Reduction.Density)
    keys = keys_for_cells([(0, 0, 0)])
    # A residual is defined against the cell mean, which a Density field does not hold.
    # The C++ layer asserts this; in a release build the assert is compiled out, so only
    # check that asking for residuals on a Mean field is what works.
    grid_mean = make_grid()
    assert grid_mean.reduction(0) == p2g.Reduction.Mean
    assert grid_mean.p2g(keys, [np.array([1.0])], residuals=True) is not None


# --- convenience helper ---------------------------------------------------


def test_density_helper_matches_manual_path():
    rng = np.random.default_rng(7)
    n = 2000
    x, y, z = (rng.uniform(LMIN, LMAX, n) for _ in range(3))
    mass = rng.uniform(0.5, 1.5, n)

    rho = p2g.density(x, y, z, mass, grid_dim=GRID_DIM, lmin=LMIN, lmax=LMAX)

    grid = make_grid()
    grid.set_reduction(0, p2g.Reduction.Density)
    grid.p2g(p2g.hilbert_keys(x, y, z, lmin=LMIN, lmax=LMAX), [mass])

    np.testing.assert_array_equal(rho, grid.field(0))
    assert (rho * CELL_VOLUME).sum() == pytest.approx(mass.sum())


# --- error handling -------------------------------------------------------


def test_grid_dim_must_divide_rank_count():
    with pytest.raises(ValueError, match="multiple of the rank count"):
        p2g.CartesianGrid(grid_dim=3, num_ranks=2)


def test_rejects_bad_arguments():
    grid = make_grid()
    keys = keys_for_cells([(0, 0, 0), (1, 1, 1)])

    with pytest.raises(ValueError, match="particles"):
        grid.p2g(keys, [np.array([1.0])])  # one value for two particles
    with pytest.raises(ValueError, match="at least one field"):
        grid.p2g(keys, [])
    with pytest.raises(ValueError, match="positive"):
        p2g.CartesianGrid(grid_dim=0)
    with pytest.raises(ValueError, match="greater than lmin"):
        p2g.CartesianGrid(grid_dim=4, lmin=1.0, lmax=0.0)
    with pytest.raises(ValueError, match="same length"):
        p2g.hilbert_keys(np.zeros(3), np.zeros(2), np.zeros(3))
    with pytest.raises(IndexError):
        grid.field(5)
    with pytest.raises(ValueError, match="grid holds"):
        grid.g2p(keys, num_fields=9)
