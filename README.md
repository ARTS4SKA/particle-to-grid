# particle-to-grid

Transfer particle data from an N-body or SPH simulation onto a uniform Cartesian grid, and back
again. Reads HDF5/H5Part or TIPSY checkpoints, distributes particles across MPI ranks along a
Hilbert curve, deposits them, and writes text or parallel HDF5. Usable as a command-line tool, a
C++ header, or a Python module; runs on CPU (MPI + OpenMP) and on NVIDIA GPUs when CUDA is found.

**P2G** deposits particle quantities into cells; **G2P** reads cell values back onto particles.

## How the transfer works

**Two assignment schemes**, chosen with `--interpolation` or by calling a different method:

- **`ngp`** (default) — each particle lands in the single cell containing it. Needs only the
  particles' Hilbert keys.
- **`sph`** — each particle is spread over the cells within its kernel support, using the Monaghan
  M4 cubic spline with support radius `2h`. Needs positions and a smoothing length per particle, and
  gives a field whose resolution follows the particles' own `h`.

SPH weights are normalised per particle (`Σ_c w_pc = 1`), so mass is conserved exactly whatever the
kernel normalisation or stencil truncation. Two cases that would otherwise lose mass: when `h` is
below the cell size no cell center falls inside the support, so the payload goes to the particle's
own cell (the `ngp` result — common in cosmology, where halo `h` is far below a cell); and a support
wider than `--maxSupportCells` is truncated and renormalised, equivalent to shrinking `h`. Both
schemes wrap periodically. Both have a CUDA path; host and device agree to a few ulp rather than
bitwise, since atomics change the summation order.

**Two reductions**, per field:

| Reduction | `ngp` | `sph` | Use for |
| --- | --- | --- | --- |
| `Density` | `Σ mₚ / V_cell` | `Σ mₚwₚ / V_cell` | **extensive** quantities — mass |
| `Mean` | `(Σ vₚ) / n` | `Σ vₚwₚ / Σ wₚ` | **intensive** quantities — temperature, velocity |

Mass must be summed, not averaged: averaging gives the *mean particle mass*, the same constant in
every occupied cell for equal-mass particles. The CLI applies `Density` to mass and `Mean` to extra
fields, and prints a particle-vs-grid mass check every run.

**Distribution.** The mesh is split into z-slabs, rank `r` owning planes
`k ∈ [r·slab, (r+1)·slab)` with `slab = gridDim/numRanks`; contributions to another rank's cells go
out via `MPI_Alltoallv`. So **`gridDim` must be a multiple of the rank count** — the CLI rounds up,
the APIs raise. All grid methods are collective: every rank calls them in the same order.

**The round trip is lossy.** When several particles share a cell, one number stands for all of them,
so G2P hands each particle the cell's value. That is inherent: an exact inverse needs as many cells
as particles. For an exact round trip ask P2G for **residuals**, the per-particle deviation
`rₚ = vₚ − mean(cell(p))`; G2P then returns `mean + residual`. A cell with `n` particles holds one
mean plus `n−1` independent residuals, so `(grid, residuals)` is a change of basis, not extra
information. Residuals need every field on `Mean`.

## Building

| Requirement | |
| --- | --- |
| C++20 compiler, CMake ≥ 3.22, MPI, OpenMP | required |
| HDF5 (parallel preferred) | optional — HDF5/H5Part input and output |
| CUDA ≥ 11.6 | optional — builds a second GPU executable |
| Python ≥ 3.8, pybind11, numpy | optional — builds the Python module |

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Options: `BUILD_TESTING` (on), `BUILD_PYTHON_BINDINGS` (on, skipped with a message if Python headers
or pybind11 are missing), `ENABLE_H5PART` (on, auto-disabled without HDF5), `GPU_DIRECT` (off,
CUDA-aware MPI).

This gives `build/main/src/particle-to-grid`, plus `particle-to-grid-cuda` with CUDA and
`build/main/python/particle_to_grid*.so` with the bindings. The module has no install rule — point
`PYTHONPATH` at its directory. To target a virtualenv or conda environment:

```bash
pip install pybind11 numpy
cmake -B build -DPython_EXECUTABLE=$(which python)
cmake --build build -j
```

## Command-line usage

```bash
build/main/src/particle-to-grid --checkpoint <path> [options]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--checkpoint <path>` | *required* | input checkpoint |
| `--checkpoint-type <t>` | `hdf5` | `hdf5` or `tipsy` |
| `--stepNo <n>` | `0` | step index inside an HDF5 file |
| `--lbox <L>` | — | box size; **required for TIPSY**, which stores positions in `[-0.5, 0.5)` |
| `--rhoCrit <rho>` | — | critical density, for TIPSY mass rescaling |
| `--gridSize <n>` | `2^ceil(log2(∛N))` | cells per dimension; rounded up to a multiple of the rank count |
| `--field <name>[,name]` | — | extra particle fields to transfer, e.g. `temp,vx` |
| `--interpolation <m>` | `ngp` | `ngp` or `sph` (the latter uses the checkpoint's `h` field) |
| `--maxSupportCells <n>` | `8` | caps the SPH stencil half-width, in cells |
| `--output-format <fmt>` | `text` | `text` or `hdf5` |
| `--output <path>` | `density` | output base name |
| `--no-output` | — | compute but write nothing (useful for timing) |

Density on a 128³ grid from an HDF5 checkpoint:

```bash
build/main/src/particle-to-grid --checkpoint data/turb_50.h5 --gridSize 128
# writes density.txt
```

On 4 ranks, with temperature alongside the density, as parallel HDF5:

```bash
mpirun -n 4 build/main/src/particle-to-grid \
    --checkpoint data/turb_50.h5 --gridSize 128 \
    --field temp --output-format hdf5 --output fields
# writes fields.h5 containing datasets "density" and "temp"
```

SPH interpolation instead of nearest-cell, which needs an `h` field in the checkpoint:

```bash
mpirun -n 4 build/main/src/particle-to-grid \
    --checkpoint data/turb_50.h5 --gridSize 32 --interpolation sph
# same total mass as ngp, but a smoother field: peak density 1.01 instead of 2.10
```

A TIPSY checkpoint, where the box size must be supplied:

```bash
mpirun -n 4 build/main/src/particle-to-grid \
    --checkpoint CDM_6Mpc_2048.00001 --checkpoint-type tipsy \
    --lbox 6 --gridSize 256
```

On the GPU build, same arguments:

```bash
mpirun -n 4 build/main/src/particle-to-grid-cuda --checkpoint data/turb_50.h5 --gridSize 128
```

Each run reports timings and mass conservation:

```
Total particles: 125000
Total mass: particles=1 grid=1 (relative difference 0)
Timing: read=0.0090 s  sync=0.3263 s  p2g=0.0749 s  write=0.0030 s  total=0.4133 s
```

**Output.** Text writes one file per field — `<output>.txt` for the density, `<name>.txt` for each
extra field — each line `index value`, the index running over the whole grid as
`i + j·gridDim + k·gridDim²` with rank slabs concatenated in rank order. HDF5 writes a single
`<output>.h5` in parallel, one dataset per field under `Step#0`, with `gridDim`, `numRanks`, `Lmin`
and `Lmax` attributes.

## C++ API

Header-only, in `main/src/mesh.hpp`, namespace `p2g`. Add `main/src` and the cstone include
directory to your include path.

```cpp
#include "mesh.hpp"

// gridDim^3 cells spanning [-0.5, 0.5] in each direction.
p2g::CartesianGrid<double> grid(rank, numRanks, /*gridDim=*/128, -0.5, 0.5);

grid.ensureNumFields(2);
grid.setReduction(0, p2g::Reduction::Density);   // mass -> mass density
grid.setReduction(1, p2g::Reduction::Mean);      // temperature -> cell average

std::vector<std::vector<double>*> fields{&mass, &temperature};
grid.p2g(keys, fields);                          // keys: Hilbert keys, one per particle

const std::vector<double>& rho = grid.grid_fields_[0];   // this rank's slab
```

SPH takes positions and smoothing lengths instead of keys, and runs on the GPU in the CUDA build
(`p2gSphCpu` has the same signature and always runs on the host):

```cpp
grid.setReduction(0, p2g::Reduction::Density);
grid.p2gSph(x, y, z, h, fields, /*maxSupportCells=*/8, /*periodic=*/true);
```

Reading values back onto particles, plain and with residuals:

```cpp
std::vector<double> outMass(n), outTemp(n);
std::vector<std::vector<double>*> out{&outMass, &outTemp};
grid.g2p(keys, out);        // each particle receives its cell's value
```

```cpp
std::vector<double> resTemp(n);
std::vector<std::vector<double>*> tempOnly{&temperature}, residuals{&resTemp};

grid.p2g(keys, tempOnly, residuals);  // grid holds cell means, resTemp the deviations
grid.g2p(keys, out, residuals);       // out == temperature, to rounding
```

Keys normally come from `cstone::Domain::sync`, which also distributes the particles, but can be
computed directly:

```cpp
#include "cstone/sfc/sfc.hpp"

cstone::Box<double> box(-0.5, 0.5, cstone::BoundaryType::periodic);
for (size_t i = 0; i < n; ++i)
    keys[i] = cstone::sfc3D<cstone::SfcKind<p2g::KeyType>>(x[i], y[i], z[i], box);
```

After `domain.sync` the arrays also hold halo particles copied from neighbouring ranks, whose field
values are not filled in. Restrict the transfer to `[domain.startIndex(), domain.endIndex())`, or
cell averages are diluted on more than one rank.

## Python API

```bash
pip install -r main/python/requirements.txt
cmake -B build -DPython_EXECUTABLE=$(which python) && cmake --build build -j
export PYTHONPATH=$PWD/build/main/python
```

Needs numpy at runtime; pybind11 and the Python headers to build; pytest and mpi4py for the tests;
h5py to read checkpoints from Python. The module wraps the grid, not the CLI pipeline: the
expectation is that Python already holds particle arrays (from h5py, or a coupled solver) and wants
a grid back as numpy.

Density in one call:

```python
import numpy as np
import particle_to_grid as p2g

rho = p2g.density(x, y, z, mass, grid_dim=128, lmin=-0.5, lmax=0.5)
print(rho.shape, rho.min(), rho.max())          # this rank's slab
```

Several fields, each with its own reduction:

```python
keys = p2g.hilbert_keys(x, y, z, lmin=-0.5, lmax=0.5)

grid = p2g.CartesianGrid(grid_dim=128, lmin=-0.5, lmax=0.5)   # rank/size from MPI_COMM_WORLD
grid.ensure_num_fields(2)
grid.set_reduction(0, p2g.Reduction.Density)
grid.set_reduction(1, p2g.Reduction.Mean)
grid.p2g(keys, [mass, temperature])

rho = grid.field(0)                      # numpy array, this rank's slab
total_mass = (rho * grid.cell_volume).sum()
```

SPH interpolation:

```python
sph_grid = p2g.CartesianGrid(grid_dim=128, lmin=-0.5, lmax=0.5)
sph_grid.set_reduction(0, p2g.Reduction.Density)
sph_grid.p2g_sph(x, y, z, h, [mass])        # positions and h, not keys

rho = sph_grid.field(0)                      # smoother than p2g, same total mass
p2g.sph_kernel(0.01, 0.02)                   # the kernel itself; scalars or arrays
```

Grid back to particles. Residuals make it exact, and require `Reduction.Mean` — asking for them on
a `Density` field raises `ValueError`:

```python
cell_mass, cell_temp = grid.g2p(keys, num_fields=2)     # lossy: each particle gets its cell value
```

```python
mean_grid = p2g.CartesianGrid(grid_dim=128, lmin=-0.5, lmax=0.5)   # fields default to Mean
residuals = mean_grid.p2g(keys, [temperature], residuals=True)
recovered, = mean_grid.g2p(keys, residuals=residuals)
assert np.allclose(recovered, temperature, rtol=1e-14)             # exact round trip
```

Importing the module initialises MPI only if nothing else has, and finalises only what it started,
so it cooperates with mpi4py:

```python
from mpi4py import MPI              # optional; import before particle_to_grid
import particle_to_grid as p2g

assert p2g.mpi_rank() == MPI.COMM_WORLD.Get_rank()
```

```bash
mpiexec -n 4 python main/python/example.py
```

`main/python/example.py` is a runnable version of all of the above.

### Writing a grid to HDF5

`main/python/tests/test_cdm_density.py` is a worked end-to-end case: it reads the
`scripts/CDM_6Mpc_2048.00001` TIPSY checkpoint (16.7M dark-matter particles), deposits the mass, and
writes `cdm_density_<grid_dim>.h5` with the density as one `(k, j, i)`-ordered dataset, bit-identical
to the C++ CLI's output for the same settings.

```bash
# the small case runs by default and is wired into ctest
pytest main/python/tests/test_cdm_density.py

# the large cases are opt-in, and skip themselves if the machine cannot take them
P2G_HEAVY=1 P2G_OUT_DIR=/scratch/me pytest main/python/tests/test_cdm_density.py -v -s

# also runnable directly, under MPI if you like
mpiexec -n 4 python main/python/tests/test_cdm_density.py 1024
```

`P2G_HEAVY=1` enables the 1024 and 2048 cases, `P2G_FORCE_HEAVY=1` runs them even when the resource
check says they will not fit, `P2G_OUT_DIR` sets the output directory, `P2G_LBOX` the box size in
Mpc (default 6.0).

**Plan for the memory.** A `Density` field costs `grid_dim³ × 8` bytes in RAM and again on disk, and
splitting across ranks on one node does not reduce the total; a `Mean` field adds 4 bytes per cell.
Measured peak RSS, single rank: **128** 0.6 GiB, **512** 1.7 GiB, **1024** 8.6 GiB, **2048** ~65 GiB
(the output file is 16 MB / 1.1 GB / 8.6 GB / 69 GB). At these sizes use `grid.field_view(f)` rather
than `grid.field(f)`: it borrows the grid's buffer instead of copying it.

### Reference

| Name | Description |
| --- | --- |
| `mpi_rank()`, `mpi_size()` | this process's position in `MPI_COMM_WORLD` |
| `hilbert_keys(x, y, z, lmin, lmax)` | Hilbert keys for particle positions |
| `key_to_cell(key, grid_dim)` / `slab_rank(k, grid_dim, num_ranks)` | cell of a key / rank owning plane `k` |
| `density(x, y, z, mass, grid_dim, lmin, lmax)` | positions and masses to a density slab in one call |
| `sph_kernel(r, h)` | the M4 cubic spline, support `2h`, normalised; scalars or arrays |
| `Reduction.Density`, `Reduction.Mean` | per-field reduction |
| `CartesianGrid(grid_dim, lmin, lmax, rank, num_ranks)` | the grid; `rank`/`num_ranks` default to `MPI_COMM_WORLD` |
| `grid.p2g(keys, fields, residuals=False)` | deposit; returns residual arrays when asked |
| `grid.p2g_sph(x, y, z, h, fields, max_support_cells=8, periodic=True)` | SPH deposition |
| `grid.g2p(keys, num_fields=None, residuals=None)` | gather; returns a list of arrays |
| `grid.field(f)` / `grid.field_view(f)` | this rank's slab, copied / without copying |
| `grid.ensure_num_fields(n)`, `set_reduction`, `reduction` | field setup |
| `grid.cell_volume`, `local_size`, `grid_dim`, `rank`, `num_ranks`, `lmin`, `lmax` | geometry |

## Testing

```bash
cd build
ctest                                   # everything
ctest -R RoundTrip --output-on-failure   # one suite
mpirun -n 4 main/test/mesh_mpi --gtest_filter='CartesianGridTest.*'   # one gtest case
```

| Test | Ranks | Covers |
| --- | --- | --- |
| `meshTest` | 4 | slab and key arithmetic, grid construction |
| `InterpolationTest` | 1 | P2G/G2P semantics, density reduction, residuals, SPH kernel |
| `FileIOTest`, `FileIOTestMultiRank` | 1, 4 | HDF5 and TIPSY readers, HDF5 grid output |
| `RoundTripTest`, `RoundTripTestMultiRank` | 1, 4 | round trips, mass conservation, SPH on real data |
| `SphCudaTest`, `SphCudaTestMultiRank` | 1, 4 | SPH on GPU against the CPU path (CUDA builds only) |
| `PythonBindingsTest`, `…MultiRank` | 1, 4 | the Python API |
| `PythonCdmDensityTest` | 1 | TIPSY checkpoint to HDF5 (small grid; skips without the data file) |

The Python tests are registered only when the configured interpreter has numpy, and the serial one
needs pytest. `cstone` ships its own suites, built by the same top-level project.

## Repository layout

```
main/src/          the tool: CLI, pipeline, grid (mesh.hpp) and CUDA backend (mesh.cu)
main/test/         GTest suites
main/python/       pybind11 module, pytest suite, example
cstone/            vendored Cornerstone octree: SFC keys, distributed domain, halos
extern/io/         file I/O abstraction (HDF5/H5Part, TIPSY)
extern/h5part/     vendored H5Part
scripts/           Python helpers for inspecting and converting TIPSY files
data/              small test fixture
```

## Limitations

- **Two schemes only**, `ngp` and `sph`, both on CPU and GPU; CIC and TSC are not implemented.
- **`ngp` extra fields are unweighted means**; `sph` weights them by the kernel. Mass weighting is
  not implemented for either.
- **SPH needs a meaningful `h`.** With a constant softening length below the cell size every
  particle falls back to its own cell and `sph` reduces to `ngp`.
- **`gridDim` must be a multiple of the rank count**, which also caps usable ranks at `gridDim`.
- **The TIPSY reader reads only dark-matter particles**; gas and star records are skipped.
- **The Python API copies arrays** in and out, since the C++ interface takes `std::vector`.
- The CLI and Python module work in `double`; the C++ grid is templated on the value type.

## License

MIT — see [LICENSE](LICENSE). The vendored `cstone` and `H5Part` components carry their own licenses.
