#include "p2g/run.hpp"
#include "p2g/utils.hpp"
#include "mesh.hpp"
#include "ifile_io_impl.h"
#include "cstone/domain/domain.hpp"

#ifdef SPH_EXA_HAVE_H5PART
#include "h5part_wrapper.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace sphexa;

namespace p2g {

using T = double;

namespace {

// Fixed number of scalar field slots (mass + extras). All slots are always passed to
// domain.sync so there is a single fixed-arity call; unused slots stay zero.
constexpr size_t MAX_FIELDS = 9;

// Next power of two >= floor(cbrt(N)).
int defaultGridDim(size_t numParticles)
{
    const size_t simDim = std::cbrt(numParticles);
    return static_cast<int>(std::pow(2, std::ceil(std::log2(simDim))));
}

// Smallest multiple of numRanks that is >= max(gridDim, numRanks), as the z-slab split requires.
int slabCompatibleGridDim(int gridDim, int numRanks, int rank)
{
    int adjusted = std::max(gridDim, numRanks);
    adjusted     = (adjusted + numRanks - 1) / numRanks * numRanks;
    if (adjusted != gridDim && rank == 0)
        std::cerr << "gridDim " << gridDim << " is not a positive multiple of numRanks " << numRanks
                  << "; using " << adjusted << ".\n";
    return adjusted;
}

void writeText(const CartesianGrid<T>& grid, const std::vector<std::string>& names, int rank, int numRanks)
{
    const int localCount = static_cast<int>(grid.localSize());
    for (size_t f = 0; f < names.size(); ++f)
    {
        std::vector<T> global(rank == 0 ? grid.localSize() * numRanks : 0);
        MPI_Gather(grid.grid_fields_[f].data(), localCount, MPI_DOUBLE, global.data(), localCount, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        if (rank != 0) continue;

        const std::string fname = names[f] + ".txt";
        std::ofstream     file(fname);
        for (size_t i = 0; i < global.size(); ++i)
            file << i << " " << std::scientific << global[i] << "\n";
        std::cout << "Saved " << fname << std::endl;
    }
}

#ifdef SPH_EXA_HAVE_H5PART
void writeHdf5(const CartesianGrid<T>& grid, const std::string& path, const std::vector<std::string>& fieldNames,
               int rank, int numRanks)
{
    H5PartFile* h5File = fileutils::openH5Part(path, H5PART_WRITE | H5PART_VFD_MPIIO_IND, MPI_COMM_WORLD);
    if (!h5File) throw std::runtime_error("Failed to open HDF5 output file: " + path);

    H5PartSetStep(h5File, 0);
    H5PartSetNumParticles(h5File, static_cast<h5part_int64_t>(grid.localSize()));

    int    gridDim = grid.gridDim_;
    double lmin    = grid.Lmin_;
    double lmax    = grid.Lmax_;
    fileutils::writeH5PartStepAttrib(h5File, "gridDim", &gridDim, 1);
    fileutils::writeH5PartStepAttrib(h5File, "numRanks", &numRanks, 1);
    fileutils::writeH5PartStepAttrib(h5File, "Lmin", &lmin, 1);
    fileutils::writeH5PartStepAttrib(h5File, "Lmax", &lmax, 1);

    for (size_t f = 0; f < fieldNames.size(); ++f)
        fileutils::writeH5PartField(h5File, fieldNames[f], grid.grid_fields_[f].data());

    H5PartCloseFile(h5File);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::cout << "Saved " << fieldNames.size() << " field(s) to " << path << std::endl;
}
#endif

} // namespace

void run(const Config& config, int rank, int numRanks)
{
    const bool tipsy = config.checkpoint_type == CheckpointType::Tipsy;

    std::unique_ptr<IFileReader> reader = tipsy ? makeTipsyReader(MPI_COMM_WORLD) : makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(config.checkpoint_path, tipsy ? 0 : config.step_no, FileMode::collective);

    const size_t numParticles = reader->globalNumParticles();
    const size_t localNum     = reader->localNumParticles();
    if (rank == 0) std::cout << "Total particles: " << numParticles << std::endl;

    const size_t numExtra = std::min(config.extra_field_names.size(), MAX_FIELDS - 1);
    if (config.extra_field_names.size() > numExtra && rank == 0)
        std::cerr << "Warning: only " << numExtra << " extra fields supported; ignoring the rest.\n";
    const size_t numFields = 1 + numExtra;

    std::vector<T> x(localNum), y(localNum), z(localNum), h(localNum, T(0));
    std::vector<T> scratch1(localNum), scratch2(localNum), scratch3(localNum);
    std::array<std::vector<T>, MAX_FIELDS> fields;
    for (auto& f : fields)
        f.assign(localNum, T(0));

    Timer timer(std::cout);
    timer.start();

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    try { reader->readField("h", h.data()); } catch (...) {}
    try { reader->readField("m", fields[0].data()); }
    catch (...) { throw std::runtime_error("mass field 'm' not available"); }
    for (size_t i = 0; i < numExtra; ++i)
    {
        try { reader->readField(config.extra_field_names[i], fields[1 + i].data()); }
        catch (...)
        {
            if (rank == 0)
                std::cerr << "Extra field '" << config.extra_field_names[i] << "' not found; using zeros.\n";
        }
    }
    reader->closeStep();

    // TIPSY positions are in [-0.5, 0.5); map them to [0, lbox) and masses to physical units.
    if (tipsy)
    {
        const T massScale = config.rho_crit > 0.0 ? config.rho_crit * config.lbox * config.lbox * config.lbox : T(1);
        for (size_t i = 0; i < localNum; ++i)
        {
            x[i] = (x[i] + T(0.5)) * config.lbox;
            y[i] = (y[i] + T(0.5)) * config.lbox;
            z[i] = (z[i] + T(0.5)) * config.lbox;
            fields[0][i] *= massScale;
        }
    }
    const double boxMin = tipsy ? 0.0 : -0.5;
    const double boxMax = tipsy ? config.lbox : 0.5;

    float t_read = timer.elapsed("Checkpoint read");

    const size_t               bucketSizeFocus = 64;
    const size_t               bucketSize      = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    const float                theta           = 1.0f;
    std::vector<KeyType>       keys(localNum);
    cstone::Box<double>        box(boxMin, boxMax, cstone::BoundaryType::periodic);
    cstone::Domain<KeyType, T, cstone::CpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h,
                std::tie(fields[0], fields[1], fields[2], fields[3], fields[4], fields[5], fields[6], fields[7],
                         fields[8]),
                std::tie(scratch1, scratch2, scratch3));

    float t_sync = timer.elapsed("Sync");

    const int gridDim = slabCompatibleGridDim(
        config.grid_size > 0 ? config.grid_size : defaultGridDim(numParticles), numRanks, rank);

    // Drop halo particles: their fields are not synced, and they belong to other ranks anyway.
    auto ownedOnly = [&](auto& v)
    {
        v.erase(v.begin() + domain.endIndex(), v.end());
        v.erase(v.begin(), v.begin() + domain.startIndex());
    };
    ownedOnly(keys);
    std::vector<std::vector<T>*> field_ptrs;
    for (size_t f = 0; f < numFields; ++f)
    {
        ownedOnly(fields[f]);
        field_ptrs.push_back(&fields[f]);
    }

    CartesianGrid<T> grid(rank, numRanks, gridDim, boxMin, boxMax);
    grid.p2g(keys, field_ptrs);

    float t_p2g   = timer.elapsed("P2G rasterization");
    float t_write = 0.f;

    if (config.write_output)
    {
        if (config.output_format == OutputFormat::Text)
        {
            std::vector<std::string> fileNames{config.output_path};
            fileNames.insert(fileNames.end(), config.extra_field_names.begin(),
                             config.extra_field_names.begin() + numExtra);
            writeText(grid, fileNames, rank, numRanks);
        }
        else
        {
#ifdef SPH_EXA_HAVE_H5PART
            std::vector<std::string> fieldNames{"density"};
            fieldNames.insert(fieldNames.end(), config.extra_field_names.begin(),
                              config.extra_field_names.begin() + numExtra);
            writeHdf5(grid, config.output_path + ".h5", fieldNames, rank, numRanks);
#else
            throw std::runtime_error("HDF5 output requested but H5Part not compiled in. Use --output-format text.");
#endif
        }
        t_write = timer.elapsed("Output write");
    }

    if (rank == 0)
    {
        std::cout << "Timing: read=" << std::fixed << std::setprecision(4) << t_read << " s  sync=" << t_sync
                  << " s  p2g=" << t_p2g << " s  write=" << t_write << " s  total=" << timer.totalElapsed() << " s"
                  << std::endl;
    }
}

} // namespace p2g
