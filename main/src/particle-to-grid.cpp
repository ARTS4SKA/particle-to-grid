#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>

#include "mesh.hpp"
#include "utils.hpp"
#include "arg_parser.hpp"
#include "ifile_io_impl.h"
#include "cstone/domain/domain.hpp"

using namespace sphexa;

void printParticleToGridHelp(char* binName, int rank);
using MeshType = double;

int main(int argc, char** argv)
{
    auto [rank, numRanks] = initMpi();
    const ArgParser parser(argc, (const char**)argv);

    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help"))
    {
        printParticleToGridHelp(argv[0], rank);
        return exitSuccess();
    }

    using KeyType        = uint64_t;
    using CoordinateType = double;

    using Domain = cstone::Domain<KeyType, CoordinateType, cstone::CpuTag>;

    const std::string initFile       = parser.get("--checkpoint");
    const std::string checkpointType = parser.get("--checkpoint-type", std::string("hdf5"));
    int               stepNo         = parser.get("--stepNo", 0);
    int               gridSizeArg    = parser.get("--gridSize", 0);
    double            lbox           = parser.get("--lbox", 0.0);
    double            rhoCrit        = parser.get("--rhoCrit", 0.0); // critical density in Msun/(h Mpc)^3
    
    Timer timer(std::cout);

    // select file format and create appropriate reader
    std::unique_ptr<IFileReader> reader;
    std::string                  typeLower = checkpointType;
    std::transform(typeLower.begin(), typeLower.end(), typeLower.begin(), ::tolower);

    if (typeLower == "tipsy")
    {
        // TIPSY format
        reader = makeTipsyReader(MPI_COMM_WORLD);
        reader->setStep(initFile, 0, FileMode::collective); // TIPSY doesn't have steps
    }
    else if (typeLower == "hdf5")
    {
        // HDF5 format
        reader = makeH5PartReader(MPI_COMM_WORLD);
        reader->setStep(initFile, stepNo, FileMode::collective);
    }
    else
    {
        if (rank == 0)
        {
            std::cerr << "Unknown --checkpoint-type '" << checkpointType
                      << "'. Supported types are 'hdf5' and 'tipsy'.\n";
        }
        return exitSuccess();
    }

    size_t numParticles = reader->globalNumParticles(); // total number of particles in the simulation
    // print numParticles on rank 0
    if (rank == 0)
    {
        std::cout << "Total number of particles in the simulation: " << numParticles << std::endl;
    }
    size_t simDim       = std::cbrt(numParticles);      // dimension of the simulation

    std::vector<double> x(reader->localNumParticles());
    std::vector<double> y(reader->localNumParticles());
    std::vector<double> z(reader->localNumParticles());
    std::vector<double> h(reader->localNumParticles(), 0.0); // interaction radius, initialized to 0
    std::vector<double> mass(reader->localNumParticles(), 1.0); // default to 1.0 if not available
    std::vector<double> scratch1(x.size());
    std::vector<double> scratch2(x.size());
    std::vector<double> scratch3(x.size());

    timer.start();

    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    // Note: h (interaction radius) is not read from file, kept at 0.0
    
    // Try to read mass field (only available for TIPSY)
    try {
        reader->readField("mass", mass.data());
    } catch (...) {
        // Mass field not available, keep default 1.0
        std::cout << "rank " << rank << " mass field not available" << std::endl;
        return exitSuccess();
    }
    
    reader->closeStep();
    
    // Transform positions and scale mass to match read_pkdgrav.py
    // Python: pos_dark = (dark[:,1:4] + 1/2) * lbox
    // Python: mass_dark = dark[:,0] * ρ_c * lbox**3
    if (lbox > 0.0 && typeLower == "tipsy")
    {
        for (size_t i = 0; i < x.size(); ++i)
        {
            // Transform from [-0.5, 0.5] to [0, lbox]
            x[i] = (x[i] + 0.5) * lbox;
            y[i] = (y[i] + 0.5) * lbox;
            z[i] = (z[i] + 0.5) * lbox;
            
            // Scale mass: mass * ρ_c * lbox^3
            if (rhoCrit > 0.0)
            {
                mass[i] = mass[i] * rhoCrit * lbox * lbox * lbox;
            }
        }
    }

    timer.elapsed("Checkpoint read");

    std::cout << "Read " << reader->localNumParticles() << " particles on rank " << rank << std::endl;

    // choose mesh dimension
    int powerDim = std::ceil(std::log(simDim) / std::log(2));
    int gridDim = (gridSizeArg > 0) ? gridSizeArg : static_cast<int>(std::pow(2, powerDim));

    // init mesh with box [0, lbox] to match read_pkdgrav.py
    double meshLmin = (lbox > 0.0 && typeLower == "tipsy") ? 0.0 : -0.5;
    double meshLmax = (lbox > 0.0 && typeLower == "tipsy") ? lbox : 0.5;
    Mesh<MeshType> mesh(rank, numRanks, gridDim, meshLmin, meshLmax);

    // create cornerstone tree and perform domain sync (production path)
    std::vector<KeyType> keys(x.size());
    size_t               bucketSizeFocus = 64;
    size_t               bucketSize      = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float                theta           = 1.0;
    cstone::Box<double>  box(meshLmin, meshLmax, cstone::BoundaryType::periodic);
    Domain               domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);

    domain.sync(keys, x, y, z, h, std::tie(mass), std::tie(scratch1, scratch2, scratch3));
    std::cout << "rank = " << rank << " numLocalParticles after sync = " << domain.nParticles() << std::endl;
    std::cout << "rank = " << rank << " numLocalParticleswithHalos after sync = " << domain.nParticlesWithHalos()
                << std::endl;
    std::cout << "rank = " << rank << " keys size after sync = " << keys.size() << std::endl;
    // std::cout << "rank = " << rank << " keys.begin = " << *keys.begin() << " keys.end = " << *keys.end() <<
    // std::endl;

    scratch1.clear();
    scratch2.clear();
    scratch3.clear();

    timer.elapsed("Sync");

#ifdef USE_CUDA
        rasterize_particles_to_mesh_cuda(mesh, keys, x, y, z, mass);
#else
        // Use binned rasterization (cell edges) to match Python's binned_statistic_dd
        // This assigns particles to bins based on which cell edge range they fall into
        mesh.rasterize_particles_to_mesh(keys, x, y, z, mass);
#endif

    std::cout << "rasterized" << std::endl;
    timer.elapsed("Rasterization");

    // Write density field to text file (flattened, one value per line)
    if (rank == 0)
    {
        std::ofstream file("density.txt");
        // Collect density from all ranks and write
        // For now, just write local density - full implementation would gather from all ranks
        for (size_t i = 0; i < mesh.dens_.size(); ++i)
        {
            file << i << " "<< std::scientific << mesh.dens_[i] << std::endl;
        }
        file.close();
        std::cout << "Saved density field to density.txt" << std::endl;
    }

    int exitCode = exitSuccess();
    
    return exitCode;
}

void printParticleToGridHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--checkpoint \t\t Input file with simulation data\n\n");
        printf("\t--checkpoint-type \t\t Checkpoint type: 'hdf5' (default) or 'tipsy'\n\n");
        printf("\t--stepNo \t\t Step number for HDF5 checkpoint file (ignored for TIPSY)\n\n");
        printf("\t--lbox \t\t\t Box size in Mpc/h (required for TIPSY to match read_pkdgrav.py)\n\n");
        printf("\t--rhoCrit \t\t Critical density in Msun/(h Mpc)^3 (required for TIPSY to match read_pkdgrav.py)\n\n");
        printf("\t--gridSize \t\t Grid dimension for Eulerian mesh (default: 2^powerDim)\n\n");
        printf("\t--direct-bin \t\t Debug mode: bypass domain sync and bin particles directly on the mesh (single rank only)\n\n");
    }
}