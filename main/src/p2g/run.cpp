#include "p2g/run.hpp"
#include "p2g/config.hpp"
#include "p2g/utils.hpp"
#include "mesh.hpp"
#include "ifile_io_impl.h"
#include "cstone/domain/domain.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace sphexa;

namespace p2g {

namespace {

template<typename T>
void rasterize_dispatch(Mesh<T>& mesh, InterpolationMethod method,
                       std::vector<KeyType>& keys,
                       std::vector<T>& x, std::vector<T>& y, std::vector<T>& z,
                       std::vector<T>& h, std::vector<T>& mass)
{
    if (method == InterpolationMethod::NearestNeighbor)
    {
#ifdef USE_CUDA
        rasterize_particles_to_mesh_cuda(mesh, keys, x, y, z, mass);
#else
        mesh.rasterize_particles_to_mesh(keys, x, y, z, mass);
#endif
    }
    else if (method == InterpolationMethod::CellAverage)
    {
#ifdef USE_CUDA
        rasterize_particles_to_mesh_cuda_cell_average(mesh, x, y, z, mass);
#else
        mesh.rasterize_particles_to_mesh_cell_average(x, y, z, mass);
#endif
    }
    else
    {
#ifdef USE_CUDA
        rasterize_particles_to_mesh_cuda_sph(mesh, x, y, z, h, mass);
#else
        mesh.rasterize_particles_to_mesh_sph(x, y, z, h, mass);
#endif
    }
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
    if (rank == 0) std::cout << "Total number of particles: " << numParticles << std::endl;
    size_t simDim = std::cbrt(numParticles);

    std::vector<T> x(reader->localNumParticles());
    std::vector<T> y(reader->localNumParticles());
    std::vector<T> z(reader->localNumParticles());
    std::vector<T> h(reader->localNumParticles(), T(0));
    std::vector<T> mass(reader->localNumParticles(), T(1));
    std::vector<T> scratch1(x.size()), scratch2(x.size()), scratch3(x.size());

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
        reader->readField("mass", mass.data());
    }
    catch (...)
    {
        if (rank == 0) std::cerr << "mass field not available" << std::endl;
        return false;
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

    timer.elapsed("Checkpoint read");
    std::cout << "Read " << reader->localNumParticles() << " particles on rank " << rank << std::endl;

    int gridDim = (config.grid_size > 0)
                      ? config.grid_size
                      : static_cast<int>(std::pow(2, std::ceil(std::log(simDim) / std::log(2))));
    double meshLmin = (config.lbox > 0.0 && typeLower == "tipsy") ? 0.0 : -0.5;
    double meshLmax = (config.lbox > 0.0 && typeLower == "tipsy") ? config.lbox : 0.5;

    Mesh<T> mesh(rank, numRanks, gridDim, meshLmin, meshLmax);

    std::vector<KeyType> keys(x.size());
    size_t bucketSizeFocus = 64;
    size_t bucketSize     = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float theta           = 1.0f;
    cstone::Box<double> box(meshLmin, meshLmax, cstone::BoundaryType::periodic);
    cstone::Domain<KeyType, T, cstone::CpuTag> domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(mass), std::tie(scratch1, scratch2, scratch3));
    scratch1.clear();
    scratch2.clear();
    scratch3.clear();
    timer.elapsed("Sync");

    rasterize_dispatch(mesh, config.interpolation, keys, x, y, z, h, mass);

    std::cout << "rasterized" << std::endl;
    timer.elapsed("Rasterization");

    if (rank == 0)
    {
        std::ofstream file("density.txt");
        for (size_t i = 0; i < mesh.dens_.size(); ++i)
            file << i << " " << std::scientific << mesh.dens_[i] << "\n";
        file.close();
        std::cout << "Saved density field to density.txt" << std::endl;
    }

    return true;
}

} // namespace p2g
