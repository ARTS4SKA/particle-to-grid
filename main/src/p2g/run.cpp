#include "p2g/run.hpp"
#include "p2g/config.hpp"
#include "p2g/utils.hpp"
#include "mesh.hpp"
#include "ifile_io_impl.h"
#include "cstone/domain/domain.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>

using namespace sphexa;

namespace p2g {

namespace {

template<typename T>
void rasterize_single(Mesh<T>& mesh, InterpolationMethod method,
                      std::vector<KeyType>& keys,
                      std::vector<T>& x, std::vector<T>& y, std::vector<T>& z,
                      std::vector<T>& h, std::vector<T>& values)
{
    if (method == InterpolationMethod::NearestNeighbor)
    {
#ifdef USE_CUDA
        rasterize_particles_to_mesh_cuda(mesh, keys, x, y, z, values);
#else
        mesh.rasterize_particles_to_mesh(keys, x, y, z, values);
#endif
    }
    else if (method == InterpolationMethod::CellAverage)
    {
#ifdef USE_CUDA
        rasterize_particles_to_mesh_cuda_cell_average(mesh, x, y, z, values);
#else
        mesh.rasterize_particles_to_mesh_cell_average(x, y, z, values);
#endif
    }
    else
    {
#ifdef USE_CUDA
        rasterize_particles_to_mesh_cuda_sph(mesh, x, y, z, h, values);
#else
        mesh.rasterize_particles_to_mesh_sph(x, y, z, h, values);
#endif
    }
}

template<typename T>
void rasterize_dispatch(Mesh<T>& mesh, InterpolationMethod method,
                        std::vector<KeyType>& keys,
                        std::vector<T>& x, std::vector<T>& y, std::vector<T>& z,
                        std::vector<T>& h, std::vector<T>& mass,
                        const std::vector<std::vector<T>*>& extraFields = {})
{
    const size_t numFields = 1u + extraFields.size();
    mesh.ensureNumFields(numFields);

    std::vector<std::vector<T>*> field_ptrs;
    field_ptrs.push_back(&mass);
    for (size_t i = 0; i < extraFields.size(); ++i)
    {
        if (!extraFields[i] || extraFields[i]->size() != mass.size())
            throw std::runtime_error("Extra field size mismatch in rasterize_dispatch");
        field_ptrs.push_back(extraFields[i]);
    }

    if (numFields == 1)
    {
        mesh.setOutputFieldIndex(0);
        rasterize_single(mesh, method, keys, x, y, z, h, mass);
        return;
    }

    // Multi-field: one particle loop, one MPI exchange for all fields
#ifdef USE_CUDA
    auto runCudaStep = [&](size_t fieldIdx, bool doReset) {
        mesh.setOutputFieldIndex(fieldIdx);
        if (method == InterpolationMethod::NearestNeighbor)
            rasterize_particles_to_mesh_cuda(mesh, keys, x, y, z, *field_ptrs[fieldIdx], false, doReset);
        else if (method == InterpolationMethod::CellAverage)
            rasterize_particles_to_mesh_cuda_cell_average(mesh, x, y, z, *field_ptrs[fieldIdx], false, doReset);
        else
            rasterize_particles_to_mesh_cuda_sph(mesh, x, y, z, h, *field_ptrs[fieldIdx], false, doReset);
    };
    runCudaStep(0, true);
    for (size_t i = 1; i < numFields; ++i)
        runCudaStep(i, false);
    mesh.performExchangeAndAccumulate(numFields);
    if (method != InterpolationMethod::SPH)
        mesh.convertMassToDensityAllFields(numFields);
#else
    if (method == InterpolationMethod::NearestNeighbor)
        mesh.rasterize_particles_to_mesh_multi(keys, field_ptrs, numFields);
    else if (method == InterpolationMethod::CellAverage)
        mesh.rasterize_particles_to_mesh_cell_average_multi(x, y, z, field_ptrs, numFields);
    else
        mesh.rasterize_particles_to_mesh_sph_multi(x, y, z, h, field_ptrs, numFields);
#endif
}

} // namespace

bool run(Config const& config, int rank, int numRanks)
{
    using T = double;
    std::string typeLower = config.checkpoint_type;
    std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(), ::tolower);

    std::unique_ptr<IFileReader> reader;
    if (typeLower == "tipsy")
    {
        reader = makeTipsyReader(MPI_COMM_WORLD);
        reader->setStep(config.checkpoint_path, 0, FileMode::collective);
    }
    else if (typeLower == "hdf5")
    {
        reader = makeH5PartReader(MPI_COMM_WORLD);
        reader->setStep(config.checkpoint_path, config.step_no, FileMode::collective);
    }
    else
    {
        if (rank == 0) std::cerr << "Unknown --checkpoint-type. Use 'hdf5' or 'tipsy'.\n";
        return false;
    }

    size_t numParticles = reader->globalNumParticles();
    if (rank == 0)
    {
        std::cout << "Configuration: checkpoint_type=" << typeLower
                  << " interpolation=" << to_string(config.interpolation) << std::endl;
        std::cout << "Total number of particles: " << numParticles << std::endl;
    }
    size_t simDim = std::cbrt(numParticles);

    const size_t localNum = reader->localNumParticles();
    std::vector<T> x(localNum), y(localNum), z(localNum);
    std::vector<T> h(localNum, T(0));
    std::vector<T> mass(localNum, T(1));
    std::vector<T> scratch1(localNum), scratch2(localNum), scratch3(localNum);

    Timer timer(std::cout);
    timer.start();

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    bool hasSmoothingLength = false;
    try
    {
        reader->readField("h", h.data());
        hasSmoothingLength = true;
    }
    catch (...) {}
    try
    {
        reader->readField("m", mass.data());
    }
    catch (...)
    {
        if (rank == 0) std::cerr << "mass field not available" << std::endl;
        return false;
    }

    // Optional extra quantities (e.g. temperature): read by name
    constexpr size_t MAX_EXTRA_FIELDS = 8;
    const size_t numExtra = std::min(config.extra_field_names.size(), MAX_EXTRA_FIELDS);
    std::array<std::vector<T>, MAX_EXTRA_FIELDS> extraFields;
    for (size_t i = 0; i < numExtra; ++i)
    {
        extraFields[i].resize(localNum, T(0));
        try
        {
            reader->readField(config.extra_field_names[i], extraFields[i].data());
        }
        catch (...)
        {
            if (rank == 0)
                std::cerr << "Extra field '" << config.extra_field_names[i] << "' not found; using zeros.\n";
            std::fill(extraFields[i].begin(), extraFields[i].end(), T(0));
        }
    }
    reader->closeStep();

    if (config.interpolation == InterpolationMethod::SPH && !hasSmoothingLength)
    {
        if (rank == 0)
            std::cerr << "SPH requires 'h' in the checkpoint. Use --interpolation nearest or cell_average.\n";
        return false;
    }

    if (config.lbox > 0.0 && typeLower == "tipsy")
    {
        for (size_t i = 0; i < x.size(); ++i)
        {
            x[i] = (x[i] + T(0.5)) * config.lbox;
            y[i] = (y[i] + T(0.5)) * config.lbox;
            z[i] = (z[i] + T(0.5)) * config.lbox;
            if (config.rho_crit > 0.0)
                mass[i] *= config.rho_crit * config.lbox * config.lbox * config.lbox;
        }
    }

    float t_read = timer.elapsed("Checkpoint read");
    std::cout << "Read " << reader->localNumParticles() << " particles on rank " << rank << std::endl;

    int gridDim = (config.grid_size > 0)
                      ? config.grid_size
                      : static_cast<int>(std::pow(2, std::ceil(std::log(simDim) / std::log(2))));
    if (gridDim < numRanks)
    {
        if (rank == 0)
            std::cerr << "gridSize " << gridDim << " < numRanks " << numRanks
                      << "; using gridDim = numRanks (each rank needs at least one z-slab)." << std::endl;
        gridDim = numRanks;
    }
    double meshLmin = (config.lbox > 0.0 && typeLower == "tipsy") ? 0.0 : -0.5;
    double meshLmax = (config.lbox > 0.0 && typeLower == "tipsy") ? config.lbox : 0.5;

    Mesh<T> mesh(rank, numRanks, gridDim, meshLmin, meshLmax);

    std::vector<KeyType> keys(x.size());
    size_t bucketSizeFocus = 64;
    size_t bucketSize     = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float theta           = 1.0f;
    cstone::Box<double> box(meshLmin, meshLmax, cstone::BoundaryType::periodic);
    cstone::Domain<KeyType, T, cstone::CpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    auto scratch = std::tie(scratch1, scratch2, scratch3);
    switch (1u + numExtra)
    {
        case 1: domain.sync(keys, x, y, z, h, std::tie(mass), scratch); break;
        case 2: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0]), scratch); break;
        case 3: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0], extraFields[1]), scratch); break;
        case 4: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0], extraFields[1], extraFields[2]), scratch); break;
        case 5: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0], extraFields[1], extraFields[2], extraFields[3]), scratch); break;
        case 6: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0], extraFields[1], extraFields[2], extraFields[3], extraFields[4]), scratch); break;
        case 7: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0], extraFields[1], extraFields[2], extraFields[3], extraFields[4], extraFields[5]), scratch); break;
        case 8: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0], extraFields[1], extraFields[2], extraFields[3], extraFields[4], extraFields[5], extraFields[6]), scratch); break;
        case 9: domain.sync(keys, x, y, z, h, std::tie(mass, extraFields[0], extraFields[1], extraFields[2], extraFields[3], extraFields[4], extraFields[5], extraFields[6], extraFields[7]), scratch); break;
        default: break;
    }
    scratch1.clear();
    scratch2.clear();
    scratch3.clear();
    float t_sync = timer.elapsed("Sync");

    std::vector<std::vector<T>*> extraFieldPtrs;
    for (size_t i = 0; i < numExtra; ++i)
        extraFieldPtrs.push_back(&extraFields[i]);
    rasterize_dispatch(mesh, config.interpolation, keys, x, y, z, h, mass, extraFieldPtrs);

    std::cout << "rasterized" << std::endl;
    float t_raster = timer.elapsed("Rasterization");

    float t_write = 0.f;
    if (rank == 0 && config.write_output)
    {
        const size_t numFields = mesh.numFields();
        for (size_t f = 0; f < numFields; ++f)
        {
            std::string fname = (f == 0) ? "density.txt" : config.extra_field_names[f - 1] + ".txt";
            std::ofstream file(fname);
            for (size_t i = 0; i < mesh.grid_fields_[f].size(); ++i)
                file << i << " " << std::scientific << mesh.grid_fields_[f][i] << "\n";
            file.close();
            std::cout << "Saved " << (f == 0 ? "density" : config.extra_field_names[f - 1]) << " to " << fname << std::endl;
        }
        t_write = timer.elapsed("Output write");
    }

    if (rank == 0)
    {
        float total = timer.totalElapsed();
        std::cout << "Timing summary: checkpoint_type=" << typeLower
                  << " interpolation=" << to_string(config.interpolation)
                  << " | read=" << std::fixed << std::setprecision(4) << t_read
                  << " s sync=" << t_sync << " s rasterize=" << t_raster
                  << " s write=" << t_write << " s total=" << total << " s" << std::endl;
    }

    return true;
}

} // namespace p2g
