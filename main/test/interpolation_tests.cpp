#include <mpi.h>
#include <gtest/gtest.h>
#include <cmath>

#include "mesh.hpp"
#include "cstone/sfc/hilbert.hpp"
#include "p2g/interpolation_method.hpp"

namespace {
// Helper: generate a Hilbert key that maps to cell (ci, cj, ck) for a given gridDim.
p2g::KeyType cellToKey(int ci, int cj, int ck, int gridDim)
{
    unsigned divisor = 1 + static_cast<unsigned>(std::pow(2, 21)) / gridDim;
    unsigned px = static_cast<unsigned>(ci) * divisor + divisor / 2;
    unsigned py = static_cast<unsigned>(cj) * divisor + divisor / 2;
    unsigned pz = static_cast<unsigned>(ck) * divisor + divisor / 2;
    return cstone::iHilbert<p2g::KeyType>(px, py, pz);
}
} // namespace

TEST(InterpolationMethod, ParseNearest)
{
    EXPECT_EQ(p2g::parseInterpolationMethod("nearest"), p2g::InterpolationMethod::NearestNeighbor);
    EXPECT_EQ(p2g::parseInterpolationMethod("Nearest"), p2g::InterpolationMethod::NearestNeighbor);
    EXPECT_EQ(p2g::parseInterpolationMethod("NEAREST"), p2g::InterpolationMethod::NearestNeighbor);
    EXPECT_EQ(p2g::parseInterpolationMethod("nearest_neighbor"), p2g::InterpolationMethod::NearestNeighbor);
    EXPECT_EQ(p2g::parseInterpolationMethod("ngp"), p2g::InterpolationMethod::NearestNeighbor);
}

TEST(InterpolationMethod, ParseSph)
{
    EXPECT_EQ(p2g::parseInterpolationMethod("sph"), p2g::InterpolationMethod::SPH);
    EXPECT_EQ(p2g::parseInterpolationMethod("SPH"), p2g::InterpolationMethod::SPH);
}

TEST(InterpolationMethod, ParseCellAverage)
{
    EXPECT_EQ(p2g::parseInterpolationMethod("cell_average"), p2g::InterpolationMethod::CellAverage);
    EXPECT_EQ(p2g::parseInterpolationMethod("cell-average"), p2g::InterpolationMethod::CellAverage);
    EXPECT_EQ(p2g::parseInterpolationMethod("average"), p2g::InterpolationMethod::CellAverage);
}

TEST(InterpolationMethod, ParseInvalid)
{
    EXPECT_THROW(p2g::parseInterpolationMethod("invalid"), std::invalid_argument);
    EXPECT_THROW(p2g::parseInterpolationMethod(""), std::invalid_argument);
}

TEST(InterpolationMethod, ToString)
{
    EXPECT_EQ(p2g::to_string(p2g::InterpolationMethod::NearestNeighbor), "nearest");
    EXPECT_EQ(p2g::to_string(p2g::InterpolationMethod::SPH), "sph");
    EXPECT_EQ(p2g::to_string(p2g::InterpolationMethod::CellAverage), "cell_average");
}

TEST(MeshInterpolation, PositionToCell)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 10, 0.0, 1.0);
    auto [i, j, k] = mesh.positionToCell(0.05, 0.25, 0.95);
    EXPECT_EQ(i, 0);
    EXPECT_EQ(j, 2);
    EXPECT_EQ(k, 9);
}

TEST(MeshInterpolation, CellCenter)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 10, 0.0, 1.0);
    double cx, cy, cz;
    mesh.cellCenter(0, 0, 0, cx, cy, cz);
    EXPECT_NEAR(cx, 0.05, 1e-12);
    EXPECT_NEAR(cy, 0.05, 1e-12);
    EXPECT_NEAR(cz, 0.05, 1e-12);
}

TEST(MeshInterpolation, NearestNeighborOneParticle)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    // Place one particle exactly at the center of cell (2,1,0)
    double cx, cy, cz;
    mesh.cellCenter(2, 1, 0, cx, cy, cz);
    std::vector<p2g::KeyType> keys = { cellToKey(2, 1, 0, 4) };
    std::vector<double> x = {cx}, y = {cy}, z = {cz};
    std::vector<double> mass = {7.0};

    mesh.rasterize_particles_to_mesh(keys, x, y, z, mass);

    // Nearest-neighbor: the only particle wins its cell, all others remain 0
    size_t idx = 2 + 1 * 4 + 0 * 16;  // i + j*gridDim + k*gridDim^2
    EXPECT_NEAR(mesh.dens()[idx], 7.0, 1e-10);
    for (size_t i = 0; i < mesh.dens().size(); ++i)
        if (i != idx) EXPECT_DOUBLE_EQ(mesh.dens()[i], 0.0);
}

TEST(MeshInterpolation, CellAverageOneParticle)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    // Place one particle in cell (1,1,1) via key
    std::vector<p2g::KeyType> keys = { cellToKey(1, 1, 1, 4) };
    std::vector<double> mass = {8.0};

    mesh.rasterize_particles_to_mesh_cell_average(keys, mass);

    // Cell average with one particle: value = 8.0 / 1 = 8.0
    size_t idx = 1 + 1 * 4 + 1 * 16;
    EXPECT_NEAR(mesh.dens()[idx], 8.0, 1e-10);

    // All other cells should be 0
    for (size_t i = 0; i < mesh.dens().size(); ++i)
        if (i != idx) EXPECT_DOUBLE_EQ(mesh.dens()[i], 0.0);
}

TEST(MeshInterpolation, SphOneParticleAtCenter)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 8, 0.0, 1.0);
    double cx, cy, cz;
    mesh.cellCenter(4, 4, 4, cx, cy, cz);

    std::vector<double> x = {cx};
    std::vector<double> y = {cy};
    std::vector<double> z = {cz};
    std::vector<double> h = {0.1};
    std::vector<double> mass = {1.0};

    mesh.rasterize_particles_to_mesh_sph(x, y, z, h, mass);

    double cellSize = 1.0 / 8.0;
    double cellVolume = cellSize * cellSize * cellSize;
    double totalMass = 0;
    for (size_t i = 0; i < mesh.dens().size(); ++i) totalMass += mesh.dens()[i] * cellVolume;
    // SPH with h=0.1, dx=0.125: support covers ~19 cells (faces+edges of center), totalMass ~5-11.
    EXPECT_GT(totalMass, 4.0);
    EXPECT_LT(totalMass, 12.0);
    // Peak density must be at the cell containing the particle
    EXPECT_GT(mesh.dens()[4 + 4*8 + 4*64], 0.0);
}

TEST(MeshInterpolation, MultiFieldCellAverageOneParticle)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    std::vector<p2g::KeyType> keys = { cellToKey(1, 1, 1, 4) };
    std::vector<double> mass = {8.0};
    std::vector<double> temp = {2.0};
    std::vector<std::vector<double>*> field_ptrs = {&mass, &temp};

    mesh.rasterize_particles_to_mesh_cell_average_multi(keys, field_ptrs, 2);

    // One particle → average = value / 1
    size_t idx = 1 + 1 * 4 + 1 * 16;
    EXPECT_NEAR(mesh.grid_fields_[0][idx], 8.0, 1e-10);
    EXPECT_NEAR(mesh.grid_fields_[1][idx], 2.0, 1e-10);
    // Only one cell has non-zero values
    double totalMass = 0, totalTemp = 0;
    for (size_t i = 0; i < mesh.grid_fields_[0].size(); ++i)
    {
        totalMass += mesh.grid_fields_[0][i];
        totalTemp += mesh.grid_fields_[1][i];
    }
    EXPECT_NEAR(totalMass, 8.0, 1e-10);
    EXPECT_NEAR(totalTemp, 2.0, 1e-10);
}

TEST(MeshInterpolation, MultiFieldSphOneParticle)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 8, 0.0, 1.0);
    double cx, cy, cz;
    mesh.cellCenter(4, 4, 4, cx, cy, cz);

    std::vector<double> x = {cx}, y = {cy}, z = {cz};
    std::vector<double> h = {0.1};
    std::vector<double> mass = {1.0};
    std::vector<double> temp = {0.5};
    std::vector<std::vector<double>*> field_ptrs = {&mass, &temp};

    mesh.rasterize_particles_to_mesh_sph_multi(x, y, z, h, field_ptrs, 2);

    double cellSize   = 1.0 / 8.0;
    double cellVolume = cellSize * cellSize * cellSize;
    double totalMass = 0, totalTemp = 0;
    for (size_t i = 0; i < mesh.grid_fields_[0].size(); ++i)
    {
        totalMass += mesh.grid_fields_[0][i] * cellVolume;
        totalTemp += mesh.grid_fields_[1][i] * cellVolume;
    }
    EXPECT_GT(totalMass, 4.0);
    EXPECT_LT(totalMass, 12.0);
    EXPECT_GT(totalTemp, 2.0);
    EXPECT_LT(totalTemp, 6.0);
    // Both fields use the same kernel weights, so the ratio must be exact
    EXPECT_NEAR(totalTemp / totalMass, 0.5, 1e-10);
    EXPECT_GT(mesh.grid_fields_[0][4 + 4*8 + 4*64], 0.0);
    EXPECT_GT(mesh.grid_fields_[1][4 + 4*8 + 4*64], 0.0);
}
