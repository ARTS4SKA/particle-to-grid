# particle-to-grid

Transfer particle data from an N-body or SPH simulation onto a uniform Cartesian grid, and back
again. Built for cosmological datasets: it reads HDF5/H5Part or TIPSY checkpoints, distributes the
particles across MPI ranks with a Hilbert space-filling curve, deposits them onto the mesh, and
writes the result as text or parallel HDF5. There is a command-line tool, a C++ API, and a Python
module.

- **P2G** (particle → grid) deposits particle quantities into cells.
- **G2P** (grid → particle) reads cell values back onto particles.

It runs on CPU (MPI + OpenMP) and, when CUDA is available, on NVIDIA GPUs.

## Contents

- [How the transfer works](#how-the-transfer-works)
- [Building](#building)
- [Command-line usage](#command-line-usage)
- [C++ API](#c-api)
- [Python API](#python-api)
- [Testing](#testing)
- [Repository layout](#repository-layout)
- [Limitations](#limitations)

## How the transfer works

### Cell assignment

Each particle carries a Hilbert key (a space-filling-curve index of its position). Decoding that key
gives the cell `(i, j, k)` the particle falls into. A particle contributes to exactly one cell —
nearest-grid-point assignment, not a spread over neighbours.

### Two reductions, and why it matters

How a field is combined inside a cell depends on the quantity:

| Reduction | Formula | Use for |
| --- | --- | --- |
| `Density` | `Σ mₚ / V_cell` | **extensive** quantities — mass |
| `Mean` | `(Σ vₚ) / n` | **intensive** quantities — temperature, velocity |

Summing mass and dividing by the cell volume conserves total mass and produces a real density.
Averaging mass instead would give the *mean particle mass*, which for equal-mass cosmological
particles is the same constant in every occupied cell and carries almost no information. The
command-line tool applies `Density` to mass and `Mean` to every extra field, and prints a
particle-vs-grid mass comparison as a conservation check.

### Distribution across ranks

The mesh is split into **z-slabs**: rank `r` owns the grid planes
`k ∈ [r·slab, (r+1)·slab)` where `slab = gridDim / numRanks`. A particle whose cell belongs to
another rank has its contribution staged and delivered with `MPI_Alltoallv`. Because of this split,
**`gridDim` must be a multiple of the number of ranks**; the CLI rounds up automatically, while the
APIs raise an error.

### The round trip is lossy — unless you keep residuals

P2G followed by G2P does not return the original values. When several particles share a cell, one
number has to stand for all of them, so each particle gets the cell's value back. This is
inherent, not a defect of the averaging: P2G is a linear map `g = W·v`, and an exact inverse needs
`rank(W) = N`, which only holds when every particle sits alone in a cell.

For an exact round trip, ask P2G for **residuals** — the per-particle deviation from its cell mean,
`rₚ = vₚ − mean(cell(p))`:

```
p2g(keys, fields, residuals)   ->  grid holds cell means, residuals hold the rest
g2p(keys, out, residuals)      ->  out = mean + residual = the original values
```

A cell with `n` particles has one mean plus `n−1` independent residuals, so the pair
`(grid, residuals)` holds exactly as many numbers as the input: it is a change of basis, not extra
information. Residuals require every field to use `Mean`, since a residual is defined against the
cell mean.

## Building

### Requirements

| Requirement | Notes |
| --- | --- |
| C++20 compiler | GCC 11+ or Clang 14+ |
| CMake | 3.22 or newer |
| MPI | OpenMPI, MPICH, … |
| OpenMP | required |
| HDF5 | optional, parallel preferred; enables HDF5/H5Part input and output |
| CUDA | optional; 11.6+ builds a second GPU executable |
| Python 3.8+, pybind11, numpy | optional; builds the Python module |

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Useful options:

| Option | Default | Effect |
| --- | --- | --- |
| `BUILD_TESTING` | `ON` | build the GTest suites |
| `BUILD_PYTHON_BINDINGS` | `ON` | build the `particle_to_grid` module (skipped with a message if Python headers or pybind11 are missing) |
| `ENABLE_H5PART` | `ON` | HDF5 input/output (auto-disabled if HDF5 is not found) |
| `GPU_DIRECT` | `OFF` | CUDA-aware MPI communication |

This produces `build/main/src/particle-to-grid`, plus `particle-to-grid-cuda` when CUDA is found and
`build/main/python/particle_to_grid*.so` when the bindings are enabled. The Python module has no
install rule — point `PYTHONPATH` at the directory holding it to import it.

To build against a specific interpreter (a virtualenv or conda environment):

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
| `--output-format <fmt>` | `text` | `text` or `hdf5` |
| `--output <path>` | `density` | output base name |
| `--no-output` | — | compute but write nothing (useful for timing) |

### Examples

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

Every run prints timings and the mass-conservation check:

```
Total particles: 125000
Total mass: particles=1 grid=1 (relative difference 0)
Timing: read=0.0090 s  sync=0.3263 s  p2g=0.0749 s  write=0.0030 s  total=0.4133 s
```

### Output

**Text** — one file per field: `<output>.txt` for the density, `<name>.txt` for each extra field.
Each line is `index value`, where the index runs over the whole grid, `i + j·gridDim +
k·gridDim²`, with the rank slabs concatenated in rank order.

**HDF5** — a single `<output>.h5` written in parallel, with one dataset per field (`density` plus
the extra field names) under `Step#0`, carrying the `gridDim`, `numRanks`, `Lmin` and `Lmax`
attributes needed to interpret the layout.

## C++ API

The grid lives in `main/src/mesh.hpp` (header-only) in namespace `p2g`. Add `main/src` and the
cstone include directory to your include path.

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

Reading values back onto particles:

```cpp
std::vector<double> outMass(n), outTemp(n);
std::vector<std::vector<double>*> out{&outMass, &outTemp};
grid.g2p(keys, out);        // each particle receives its cell's value
```

An exact round trip with residuals:

```cpp
std::vector<double> resTemp(n);
std::vector<std::vector<double>*> fields{&temperature}, residuals{&resTemp};

grid.p2g(keys, fields, residuals);    // grid holds cell means, resTemp the deviations
grid.g2p(keys, out, residuals);       // out == temperature, to rounding
```

Particle keys normally come from a `cstone::Domain::sync` call, which also distributes the
particles. They can be computed directly from positions:

```cpp
#include "cstone/sfc/sfc.hpp"

cstone::Box<double> box(-0.5, 0.5, cstone::BoundaryType::periodic);
keys[i] = cstone::sfc3D<cstone::SfcKind<p2g::KeyType>>(x[i], y[i], z[i], box);
```

Note that after `domain.sync` the arrays also contain halo particles copied from neighbouring
ranks, whose field values are not filled in. Restrict the transfer to
`[domain.startIndex(), domain.endIndex())`, or cell averages are diluted on more than one rank.

All grid methods are MPI collectives: every rank must call them, in the same order.

## Python API

```bash
pip install -r main/python/requirements.txt
cmake -B build -DPython_EXECUTABLE=$(which python) && cmake --build build -j
export PYTHONPATH=$PWD/build/main/python
```

| Package | When | Why |
| --- | --- | --- |
| pybind11, Python headers | build | compiling the module |
| numpy | runtime | every array argument and return value |
| pytest | tests | serial test suite |
| mpi4py | tests, multi-rank use | cross-rank reductions; shares the communicator |
| h5py | optional | reading checkpoints from Python |

The module wraps the grid rather than the command-line pipeline: the expectation is that Python
already holds particle arrays (from h5py, or from a coupled solver) and wants a grid back as a numpy
array.

### Density in one call

```python
import numpy as np
import particle_to_grid as p2g

rho = p2g.density(x, y, z, mass, grid_dim=128, lmin=-0.5, lmax=0.5)
print(rho.shape, rho.min(), rho.max())          # this rank's slab
```

### Several fields, each with its own reduction

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

### Grid back to particles

```python
cell_mass, cell_temp = grid.g2p(keys, num_fields=2)     # lossy: each particle gets its cell value
```

Residuals make the round trip exact. They are defined against the cell mean, so every field
involved must use `Reduction.Mean` — asking for residuals on a `Density` field raises `ValueError`:

```python
mean_grid = p2g.CartesianGrid(grid_dim=128, lmin=-0.5, lmax=0.5)   # fields default to Mean
residuals = mean_grid.p2g(keys, [temperature], residuals=True)
recovered, = mean_grid.g2p(keys, residuals=residuals)
assert np.allclose(recovered, temperature, rtol=1e-14)             # exact round trip
```

### Running under MPI

Importing the module initializes MPI only if nothing else has, and finalizes only what it started,
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

### Reference

| Name | Description |
| --- | --- |
| `mpi_rank()`, `mpi_size()` | this process's position in `MPI_COMM_WORLD` |
| `hilbert_keys(x, y, z, lmin, lmax)` | Hilbert keys for particle positions |
| `key_to_cell(key, grid_dim)` | the cell `(i, j, k)` a key falls in |
| `slab_rank(k, grid_dim, num_ranks)` | which rank owns plane `k` |
| `density(x, y, z, mass, grid_dim, lmin, lmax)` | positions and masses to a density slab in one call |
| `Reduction.Density`, `Reduction.Mean` | per-field reduction |
| `CartesianGrid(grid_dim, lmin, lmax, rank, num_ranks)` | the grid; `rank`/`num_ranks` default to `MPI_COMM_WORLD` |
| `grid.p2g(keys, fields, residuals=False)` | deposit; returns residual arrays when asked |
| `grid.g2p(keys, num_fields=None, residuals=None)` | gather; returns a list of arrays |
| `grid.field(f)` | this rank's slab of field `f` |
| `grid.ensure_num_fields(n)`, `set_reduction`, `reduction` | field setup |
| `grid.cell_volume`, `local_size`, `grid_dim`, `rank`, `num_ranks`, `lmin`, `lmax` | geometry |

## Testing

```bash
cd build
ctest                                  # everything
ctest -R RoundTrip --output-on-failure  # one suite
```

| Test | Ranks | Covers |
| --- | --- | --- |
| `meshTest` | 4 | slab and key arithmetic, grid construction |
| `InterpolationTest` | 1 | P2G/G2P semantics, density reduction, residuals |
| `FileIOTest`, `FileIOTestMultiRank` | 1, 4 | HDF5 and TIPSY readers, HDF5 grid output |
| `RoundTripTest`, `RoundTripTestMultiRank` | 1, 4 | round trips and mass conservation on real data |
| `PythonBindingsTest` | 1 | the Python API (pytest) |
| `PythonBindingsTestMultiRank` | 4 | the Python API across ranks |

The Python tests are registered only when the configured interpreter has numpy, and the serial one
needs pytest. `cstone` ships its own suites, built by the same top-level CMake project.

Individual GTest cases can be run directly:

```bash
mpirun -n 4 build/main/test/mesh_mpi --gtest_filter='CartesianGridTest.*'
```

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

- **Nearest-grid-point assignment.** A particle lands in exactly one cell. Higher-order schemes
  (CIC, TSC) spread it over neighbours and give smoother fields; they are not implemented.
- **Extra fields are unweighted means.** Mass-weighted averages are usually the better choice for
  temperature and velocity.
- **`gridDim` must be a multiple of the rank count**, which also caps usable ranks at `gridDim`.
- **The TIPSY reader reads only dark-matter particles**; gas and star records are skipped.
- **The Python API copies arrays** in and out, since the C++ interface takes `std::vector`.
- The CLI and the Python module work in `double`; the C++ grid is templated on the value type.

## License

MIT — see [LICENSE](LICENSE). The vendored `cstone` and `H5Part` components carry their own
licenses.
