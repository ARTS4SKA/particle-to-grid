// Python bindings for the particle-to-grid transfer.
//
// The module exposes the grid itself rather than the command-line pipeline: the expected use is
// that Python already holds particle arrays (from h5py, or from a coupled solver) and wants a grid
// back as a numpy array. Reading checkpoint files stays in the CLI or in h5py.
//
// Importing the module initializes MPI if nothing else has done so yet, because every grid
// operation is an MPI collective. If mpi4py is imported first, its communicator is used as-is and
// this module leaves finalization to mpi4py.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <mpi.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "cstone/sfc/sfc.hpp"
#include "mesh.hpp"

namespace py = pybind11;

using Grid    = p2g::CartesianGrid<double>;
using KeyType = p2g::KeyType;
using DArray  = py::array_t<double, py::array::c_style | py::array::forcecast>;
using KArray  = py::array_t<KeyType, py::array::c_style | py::array::forcecast>;

namespace {

bool ownsMpi = false;

void finalizeMpi()
{
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
}

void ensureMpi()
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) return;

    MPI_Init(nullptr, nullptr);
    ownsMpi = true;
    // Only finalize what we started; mpi4py cleans up after itself.
    Py_AtExit(&finalizeMpi);
}

int commRank()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int commSize()
{
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

std::vector<double> toVector(const DArray& a, const char* name)
{
    if (a.ndim() != 1) throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    return std::vector<double>(a.data(), a.data() + a.size());
}

std::vector<KeyType> toKeys(const KArray& a)
{
    if (a.ndim() != 1) throw std::invalid_argument("keys must be a 1-D array");
    return std::vector<KeyType>(a.data(), a.data() + a.size());
}

DArray toArray(const std::vector<double>& v)
{
    DArray out(static_cast<py::ssize_t>(v.size()));
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

// Copies each input array into a vector, since the C++ API works on std::vector.
std::vector<std::vector<double>> collectFields(const py::sequence& fields, size_t numParticles)
{
    if (py::len(fields) == 0) throw std::invalid_argument("at least one field is required");

    std::vector<std::vector<double>> out;
    for (size_t f = 0; f < py::len(fields); ++f)
    {
        auto arr = fields[f].cast<DArray>();
        if (static_cast<size_t>(arr.size()) != numParticles)
            throw std::invalid_argument("field " + std::to_string(f) + " has " + std::to_string(arr.size()) +
                                        " entries but there are " + std::to_string(numParticles) + " particles");
        out.push_back(toVector(arr, "field"));
    }
    return out;
}

std::vector<std::vector<double>*> pointersTo(std::vector<std::vector<double>>& storage)
{
    std::vector<std::vector<double>*> ptrs;
    for (auto& v : storage)
        ptrs.push_back(&v);
    return ptrs;
}

py::list toArrayList(const std::vector<std::vector<double>>& storage)
{
    py::list out;
    for (const auto& v : storage)
        out.append(toArray(v));
    return out;
}

// gridDim must divide evenly into the rank count: the grid is split into z-slabs of equal depth.
void checkGridDim(int gridDim, int numRanks)
{
    if (gridDim <= 0) throw std::invalid_argument("grid_dim must be positive");
    if (numRanks <= 0) throw std::invalid_argument("num_ranks must be positive");
    if (gridDim % numRanks != 0)
        throw std::invalid_argument("grid_dim (" + std::to_string(gridDim) + ") must be a multiple of the rank count (" +
                                    std::to_string(numRanks) + "), because the grid is split into z-slabs");
}

std::unique_ptr<Grid> makeGrid(int gridDim, double lmin, double lmax, std::optional<int> rank,
                               std::optional<int> numRanks)
{
    ensureMpi();
    const int r = rank.value_or(commRank());
    const int n = numRanks.value_or(commSize());
    checkGridDim(gridDim, n);
    if (lmax <= lmin) throw std::invalid_argument("lmax must be greater than lmin");
    return std::make_unique<Grid>(r, n, gridDim, lmin, lmax);
}

// Hilbert keys for particle positions, matching what cstone::Domain::sync would assign.
KArray hilbertKeys(const DArray& x, const DArray& y, const DArray& z, double lmin, double lmax)
{
    if (x.size() != y.size() || x.size() != z.size())
        throw std::invalid_argument("x, y and z must have the same length");
    if (lmax <= lmin) throw std::invalid_argument("lmax must be greater than lmin");

    const cstone::Box<double> box(lmin, lmax, cstone::BoundaryType::periodic);
    KArray                    keys(x.size());
    const double*             px = x.data();
    const double*             py_ = y.data();
    const double*             pz = z.data();
    KeyType*                  out = keys.mutable_data();

    for (py::ssize_t i = 0; i < x.size(); ++i)
        out[i] = cstone::sfc3D<cstone::SfcKind<KeyType>>(px[i], py_[i], pz[i], box);

    return keys;
}

py::object gridP2G(Grid& grid, const KArray& keysArr, const py::sequence& fields, bool residuals)
{
    auto keys    = toKeys(keysArr);
    auto storage = collectFields(fields, keys.size());
    auto ptrs    = pointersTo(storage);

    if (!residuals)
    {
        // The GIL is released only around the C++ call; touching Python objects needs it back.
        {
            py::gil_scoped_release unlock;
            grid.p2g(keys, ptrs);
        }
        return py::none();
    }

    // Checked here rather than left to the C++ assert: a wrong call from Python must raise,
    // not abort the interpreter, and the assert disappears entirely under NDEBUG.
    for (size_t f = 0; f < storage.size(); ++f)
        if (grid.reduction(f) != p2g::Reduction::Mean)
            throw std::invalid_argument("residuals need field " + std::to_string(f) +
                                        " to use Reduction.Mean, because a residual is the deviation from the "
                                        "cell mean; it currently uses Reduction.Density");

    std::vector<std::vector<double>> resStorage(storage.size(), std::vector<double>(keys.size(), 0.0));
    auto                             resPtrs = pointersTo(resStorage);
    {
        py::gil_scoped_release unlock;
        grid.p2g(keys, ptrs, resPtrs);
    }
    return toArrayList(resStorage);
}

py::list gridG2P(Grid& grid, const KArray& keysArr, std::optional<size_t> numFields,
                 const std::optional<py::sequence>& residuals)
{
    auto keys = toKeys(keysArr);

    size_t n = 0;
    if (residuals) { n = static_cast<size_t>(py::len(*residuals)); }
    else { n = numFields.value_or(grid.numFields()); }
    if (n == 0) throw std::invalid_argument("no fields to gather");
    if (n > grid.numFields())
        throw std::invalid_argument("requested " + std::to_string(n) + " fields but the grid holds " +
                                    std::to_string(grid.numFields()));

    std::vector<std::vector<double>> storage(n, std::vector<double>(keys.size(), 0.0));
    auto                             ptrs = pointersTo(storage);

    if (residuals)
    {
        auto resStorage = collectFields(*residuals, keys.size());
        auto resPtrs    = pointersTo(resStorage);
        {
            py::gil_scoped_release unlock;
            grid.g2p(keys, ptrs, resPtrs);
        }
    }
    else
    {
        py::gil_scoped_release unlock;
        grid.g2p(keys, ptrs);
    }
    return toArrayList(storage);
}

void checkFieldIndex(const Grid& grid, size_t f)
{
    if (f >= grid.numFields())
        throw std::out_of_range("field " + std::to_string(f) + " does not exist; the grid holds " +
                                std::to_string(grid.numFields()));
}

DArray gridField(const Grid& grid, size_t f)
{
    checkFieldIndex(grid, f);
    return toArray(grid.grid_fields_[f]);
}

// Zero-copy view of a field. Matters at large grid_dim, where copying a slab can double peak
// memory (a 1024^3 grid of doubles is 8 GiB). The returned array borrows the grid's buffer and
// keeps the grid alive; it is invalidated by anything that reallocates, i.e. ensure_num_fields()
// growing the field count. p2g() and g2p() overwrite in place and keep it valid.
py::array_t<double> gridFieldView(Grid& grid, size_t f)
{
    checkFieldIndex(grid, f);
    auto& field = grid.grid_fields_[f];
    return py::array_t<double>({static_cast<py::ssize_t>(field.size())}, {sizeof(double)}, field.data(),
                               py::cast(&grid));
}

// One-shot convenience: positions and masses in, this rank's slab of mass density out.
DArray densityFromPositions(const DArray& x, const DArray& y, const DArray& z, const DArray& mass, int gridDim,
                            double lmin, double lmax)
{
    if (static_cast<size_t>(mass.size()) != static_cast<size_t>(x.size()))
        throw std::invalid_argument("mass must have the same length as x, y and z");

    auto grid = makeGrid(gridDim, lmin, lmax, std::nullopt, std::nullopt);
    grid->setReduction(0, p2g::Reduction::Density);

    auto                              keys = toKeys(hilbertKeys(x, y, z, lmin, lmax));
    std::vector<std::vector<double>>  storage{toVector(mass, "mass")};
    std::vector<std::vector<double>*> ptrs = pointersTo(storage);
    {
        py::gil_scoped_release unlock;
        grid->p2g(keys, ptrs);
    }
    return toArray(grid->grid_fields_[0]);
}

} // namespace

PYBIND11_MODULE(particle_to_grid, m)
{
    m.doc() = "Particle-to-grid and grid-to-particle transfer on a distributed Cartesian mesh.";

    ensureMpi();

    py::enum_<p2g::Reduction>(m, "Reduction", "How a field's per-cell contributions are combined.")
        .value("Mean", p2g::Reduction::Mean, "Arithmetic mean over the particles in the cell (temperature, velocity).")
        .value("Density", p2g::Reduction::Density,
               "Sum of the particle values divided by the cell volume (mass -> mass density).");

    m.def("mpi_rank", [] { ensureMpi(); return commRank(); }, "Rank of this process in MPI_COMM_WORLD.");
    m.def("mpi_size", [] { ensureMpi(); return commSize(); }, "Number of processes in MPI_COMM_WORLD.");

    m.def("hilbert_keys", &hilbertKeys, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("lmin") = -0.5,
          py::arg("lmax") = 0.5,
          "Hilbert SFC keys for the given positions, as cstone::Domain::sync would assign them.");

    m.def(
        "key_to_cell",
        [](KeyType key, int gridDim)
        {
            if (gridDim <= 0) throw std::invalid_argument("grid_dim must be positive");
            int i, j, k;
            p2g::keyToCell(key, gridDim, i, j, k);
            return py::make_tuple(i, j, k);
        },
        py::arg("key"), py::arg("grid_dim"), "Grid cell (i, j, k) containing the given Hilbert key.");

    m.def(
        "slab_rank", [](int k, int gridDim, int numRanks) { return p2g::slabRank(k, gridDim, numRanks); },
        py::arg("k"), py::arg("grid_dim"), py::arg("num_ranks"), "Rank owning the z-slab that contains plane k.");

    m.def("density", &densityFromPositions, py::arg("x"), py::arg("y"), py::arg("z"), py::arg("mass"),
          py::arg("grid_dim"), py::arg("lmin") = -0.5, py::arg("lmax") = 0.5,
          "Mass density on this rank's slab, computed from particle positions and masses in one call.");

    py::class_<Grid>(m, "CartesianGrid",
                     "Uniform Cartesian mesh split into z-slabs across MPI ranks. All methods are collective.")
        .def(py::init(&makeGrid), py::arg("grid_dim"), py::arg("lmin") = -0.5, py::arg("lmax") = 0.5,
             py::arg("rank") = py::none(), py::arg("num_ranks") = py::none(),
             "Create a grid of grid_dim^3 cells spanning [lmin, lmax] in each direction. rank and num_ranks "
             "default to this process's position in MPI_COMM_WORLD.")
        .def_readonly("grid_dim", &Grid::gridDim_)
        .def_readonly("rank", &Grid::rank_)
        .def_readonly("num_ranks", &Grid::numRanks_)
        .def_readonly("lmin", &Grid::Lmin_)
        .def_readonly("lmax", &Grid::Lmax_)
        .def_property_readonly("local_size", &Grid::localSize, "Number of cells owned by this rank.")
        .def_property_readonly("cell_volume", &Grid::cellVolume)
        .def("num_fields", &Grid::numFields)
        .def("ensure_num_fields", &Grid::ensureNumFields, py::arg("n"),
             "Make room for n fields, zero-initialised; needed before set_reduction on a field index.")
        .def("set_reduction", &Grid::setReduction, py::arg("field"), py::arg("reduction"),
             "Choose how field `field` is combined per cell. Fields default to Reduction.Mean.")
        .def("reduction", &Grid::reduction, py::arg("field"))
        .def("p2g", &gridP2G, py::arg("keys"), py::arg("fields"), py::arg("residuals") = false,
             "Scatter particle fields onto the grid. `fields` is a sequence of per-particle arrays.\n"
             "With residuals=True, returns a list of per-particle residual arrays which g2p can use to\n"
             "reconstruct the inputs exactly; residuals require every field to use Reduction.Mean.")
        .def("g2p", &gridG2P, py::arg("keys"), py::arg("num_fields") = py::none(), py::arg("residuals") = py::none(),
             "Gather grid values back to particles, returning a list of arrays. Pass the residuals from\n"
             "p2g to recover the original per-particle values instead of the cell value.")
        .def("field", &gridField, py::arg("field"), "This rank's slab of field `field`, as a copy.")
        .def("field_view", &gridFieldView, py::arg("field"),
             "This rank's slab of field `field` without copying, for large grids. The array borrows\n"
             "the grid's memory and keeps the grid alive; ensure_num_fields() may invalidate it.")
        .def("__repr__",
             [](const Grid& g)
             {
                 return "<CartesianGrid grid_dim=" + std::to_string(g.gridDim_) + " rank=" + std::to_string(g.rank_) +
                        "/" + std::to_string(g.numRanks_) + " local_size=" + std::to_string(g.localSize()) + ">";
             });
}
