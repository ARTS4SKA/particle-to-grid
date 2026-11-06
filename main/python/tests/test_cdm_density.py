"""Deposit the CDM_6Mpc_2048 TIPSY checkpoint onto a grid and write it to HDF5.

Reads scripts/CDM_6Mpc_2048.00001 (16.7M dark-matter particles), runs P2G with the Density
reduction, and writes one HDF5 file per grid size: cdm_density_<grid_dim>.h5.

The 1024 and 2048 cases are large. A Density field costs grid_dim^3 * 8 bytes, in RAM and again
on disk, and MPI does not reduce the per-node total -- splitting across ranks on one machine does
not make the grid smaller. A Mean field costs a further 4 bytes per cell for the particle counts.

    grid_dim=128    ->   16 MiB      (measured peak RSS:  0.6 GiB)
    grid_dim=512    ->    1 GiB      (measured peak RSS:  1.7 GiB)
    grid_dim=1024   ->    8 GiB      (measured peak RSS:  8.6 GiB)
    grid_dim=2048   ->   64 GiB      (needs a cluster node)

So they are skipped unless explicitly enabled and the machine can take it:

    P2G_HEAVY=1 pytest main/python/tests/test_cdm_density.py -v -s

Environment variables:
    P2G_HEAVY=1        enable the 1024 and 2048 cases
    P2G_FORCE_HEAVY=1  run them even if the resource check says they will not fit
    P2G_OUT_DIR=<dir>  where the HDF5 files go (default: the current directory)
    P2G_LBOX=<L>       box size in Mpc for the coordinate scaling (default: 6.0)

Under MPI each rank reads its own contiguous share of the particles and owns a z-slab of the
grid. h5py from pip is built without MPI support, so ranks write their slabs one at a time.
"""

import os
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

import particle_to_grid as p2g

TIPSY_FILE = Path(__file__).resolve().parents[3] / "scripts" / "CDM_6Mpc_2048.00001"

# TIPSY is big-endian: a 32-byte header, then one 36-byte record per dark-matter particle.
HEADER_DTYPE = np.dtype(
    [("time", ">f8"), ("N", ">i4"), ("Dims", ">i4"), ("Ngas", ">i4"), ("Ndark", ">i4"), ("Nstar", ">i4"),
     ("pad", ">i4")]
)
DARK_DTYPE = np.dtype(
    [("mass", ">f4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4"), ("vx", ">f4"), ("vy", ">f4"), ("vz", ">f4"),
     ("eps", ">f4"), ("phi", ">f4")]
)
GAS_RECORD_SIZE = 12 * 4
LBOX = float(os.environ.get("P2G_LBOX", 6.0))

RANK, SIZE = p2g.mpi_rank(), p2g.mpi_size()


def log(message):
    if RANK == 0:
        print(message, flush=True)


# --- TIPSY reading --------------------------------------------------------


def read_header(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(HEADER_DTYPE.itemsize), dtype=HEADER_DTYPE, count=1)[0]


def read_dark_slice(path, header, first, last):
    """Particles [first, last) as float64 positions in [0, LBOX) and masses in TIPSY units."""
    offset = HEADER_DTYPE.itemsize + int(header["Ngas"]) * GAS_RECORD_SIZE + first * DARK_DTYPE.itemsize
    records = np.memmap(path, dtype=DARK_DTYPE, mode="r", offset=offset, shape=(last - first,))

    # TIPSY stores positions in [-0.5, 0.5); shift them into [0, LBOX) like p2g::run() does.
    x = (records["x"].astype(np.float64) + 0.5) * LBOX
    y = (records["y"].astype(np.float64) + 0.5) * LBOX
    z = (records["z"].astype(np.float64) + 0.5) * LBOX
    mass = records["mass"].astype(np.float64)
    del records
    return x, y, z, mass


def partition(total, rank, num_ranks):
    """Contiguous share of `total` items for this rank, matching the C++ reader's split."""
    base, extra = divmod(total, num_ranks)
    if rank < extra:
        first = (base + 1) * rank
        return first, first + base + 1
    first = (base + 1) * extra + base * (rank - extra)
    return first, first + base


# --- HDF5 writing ---------------------------------------------------------


# Planes written per call. Bounds HDF5's internal buffering: writing a whole 1024^3 slab in one
# assignment pushed peak RSS to 13 GiB for an 8 GiB grid, while blocks keep it near the grid size.
WRITE_BLOCK_BYTES = 512 * 1024**2


def write_slab(dset, slab, grid_dim, k0, depth):
    """Copy this rank's slab into the dataset in bounded blocks of whole planes."""
    plane_bytes = grid_dim * grid_dim * 8
    planes = max(1, min(depth, WRITE_BLOCK_BYTES // plane_bytes))
    block = slab.reshape(depth, grid_dim, grid_dim)
    for k in range(0, depth, planes):
        n = min(planes, depth - k)
        dset[k0 + k : k0 + k + n] = block[k : k + n]


def write_grid_hdf5(path, grid, slab, attrs):
    """Write the full grid as one (k, j, i)-ordered dataset, ranks taking turns.

    The local flat index is i + j*grid_dim + local_k*grid_dim^2, so reshaping a slab to
    (depth, grid_dim, grid_dim) gives axes (k, j, i).
    """
    grid_dim = grid.grid_dim
    depth = grid_dim // grid.num_ranks
    k0 = grid.rank * depth

    comm = None
    if SIZE > 1:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD

    if RANK == 0:
        with h5py.File(path, "w") as f:
            dset = f.create_dataset("density", shape=(grid_dim, grid_dim, grid_dim), dtype="f8")
            dset.attrs["axes"] = "k,j,i"
            for key, value in attrs.items():
                dset.attrs[key] = value
            write_slab(dset, slab, grid_dim, k0, depth)
    if comm is not None:
        comm.Barrier()
        # Serial h5py: one writer at a time, in rank order.
        for writer in range(1, SIZE):
            if RANK == writer:
                with h5py.File(path, "r+") as f:
                    write_slab(f["density"], slab, grid_dim, k0, depth)
            comm.Barrier()


# --- resource guard -------------------------------------------------------


def available_memory():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def ranks_on_this_node():
    if SIZE == 1:
        return 1
    try:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).Get_size()
    except ImportError:
        return SIZE


def check_resources(grid_dim, out_dir):
    """Return None if the run should fit, otherwise a reason to skip."""
    if os.environ.get("P2G_FORCE_HEAVY") == "1":
        return None

    cells = grid_dim**3
    grid_bytes = cells * 8
    per_rank = grid_bytes // SIZE
    # Particles this rank holds: keys plus the mass array it keeps.
    node_bytes = per_rank * ranks_on_this_node() + 512 * 1024**2

    mem = available_memory()
    if mem is not None and node_bytes > 0.8 * mem:
        return (
            f"needs ~{node_bytes / 2**30:.1f} GiB on this node but only "
            f"{mem / 2**30:.1f} GiB is available (set P2G_FORCE_HEAVY=1 to try anyway)"
        )

    free_disk = shutil.disk_usage(out_dir).free
    if grid_bytes > 0.9 * free_disk:
        return (
            f"the output file is {grid_bytes / 2**30:.1f} GiB but only "
            f"{free_disk / 2**30:.1f} GiB is free on {out_dir}"
        )
    return None


# --- the run itself -------------------------------------------------------


def run_cdm_p2g(grid_dim):
    """Read the checkpoint, deposit it, write the HDF5 file, and validate the result."""
    out_dir = Path(os.environ.get("P2G_OUT_DIR", Path.cwd()))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cdm_density_{grid_dim}.h5"

    if grid_dim % SIZE != 0:
        pytest.skip(f"grid_dim {grid_dim} is not a multiple of {SIZE} ranks")

    header = read_header(TIPSY_FILE)
    ndark = int(header["Ndark"])
    assert ndark > 0, "no dark-matter particles in the checkpoint"

    first, last = partition(ndark, RANK, SIZE)
    x, y, z, mass = read_dark_slice(TIPSY_FILE, header, first, last)
    local_mass = float(mass.sum())
    log(f"  {ndark} particles, {last - first} on rank 0, grid_dim={grid_dim}")

    keys = p2g.hilbert_keys(x, y, z, lmin=0.0, lmax=LBOX)
    del x, y, z  # keys are all that P2G needs; free ~3 arrays before allocating the grid

    grid = p2g.CartesianGrid(grid_dim=grid_dim, lmin=0.0, lmax=LBOX)
    grid.set_reduction(0, p2g.Reduction.Density)
    grid.p2g(keys, [mass])
    del keys

    # field_view avoids copying the slab, which matters at grid_dim 1024 and above.
    slab = grid.field_view(0)
    grid_mass = float(slab.sum()) * grid.cell_volume

    if SIZE > 1:
        from mpi4py import MPI

        total_particle_mass = MPI.COMM_WORLD.allreduce(local_mass, op=MPI.SUM)
        total_grid_mass = MPI.COMM_WORLD.allreduce(grid_mass, op=MPI.SUM)
    else:
        total_particle_mass, total_grid_mass = local_mass, grid_mass

    # Mass must survive the transfer.
    assert total_grid_mass == pytest.approx(total_particle_mass, rel=1e-10), (
        f"mass not conserved: grid={total_grid_mass} particles={total_particle_mass}"
    )

    write_grid_hdf5(
        out_path,
        grid,
        slab,
        {
            "grid_dim": grid_dim,
            "lbox": LBOX,
            "num_particles": ndark,
            "total_mass": total_particle_mass,
            "cell_volume": grid.cell_volume,
            "source": TIPSY_FILE.name,
        },
    )
    del slab, grid

    if SIZE > 1:
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()

    # Read the file back and check it describes the same field.
    if RANK == 0:
        with h5py.File(out_path, "r") as f:
            dset = f["density"]
            assert dset.shape == (grid_dim, grid_dim, grid_dim)
            assert dset.dtype == np.float64
            assert dset.attrs["grid_dim"] == grid_dim
            assert dset.attrs["axes"] == "k,j,i"

            # Sum a few planes rather than the whole field, which may not fit in memory.
            plane_total = sum(float(dset[k].sum()) for k in range(min(8, grid_dim)))
            assert plane_total > 0, "the stored density is all zeros"
            first_plane = dset[0]
            assert first_plane.min() != first_plane.max(), "the stored density is constant"

        size_gib = out_path.stat().st_size / 2**30
        log(f"  wrote {out_path} ({size_gib:.2f} GiB), total mass {total_particle_mass:.6g}")

    return out_path


# --- tests ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def require_checkpoint():
    if not TIPSY_FILE.exists():
        pytest.skip(f"checkpoint not found: {TIPSY_FILE}")


def heavy_or_skip(grid_dim):
    if os.environ.get("P2G_HEAVY") != "1":
        pytest.skip(f"grid_dim={grid_dim} needs {grid_dim**3 * 8 / 2**30:.0f} GiB; set P2G_HEAVY=1 to run")
    reason = check_resources(grid_dim, Path(os.environ.get("P2G_OUT_DIR", Path.cwd())))
    if reason:
        pytest.skip(f"grid_dim={grid_dim}: {reason}")


def test_cdm_density_small():
    """Same code path at a size that always fits, so the machinery stays tested."""
    grid_dim = int(os.environ.get("P2G_SMALL_GRID_DIM", 128))
    log(f"CDM P2G, grid_dim={grid_dim} (smoke test):")
    run_cdm_p2g(grid_dim)


def test_cdm_density_1024():
    heavy_or_skip(1024)
    log("CDM P2G, grid_dim=1024:")
    assert run_cdm_p2g(1024).name == "cdm_density_1024.h5"


def test_cdm_density_2048():
    heavy_or_skip(2048)
    log("CDM P2G, grid_dim=2048:")
    assert run_cdm_p2g(2048).name == "cdm_density_2048.h5"


if __name__ == "__main__":
    # Also runnable without pytest: python test_cdm_density.py [grid_dim ...]
    os.environ.setdefault("P2G_HEAVY", "1")
    for arg in sys.argv[1:] or ["128"]:
        log(f"CDM P2G, grid_dim={arg}:")
        run_cdm_p2g(int(arg))
