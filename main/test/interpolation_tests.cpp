#include <mpi.h>
#include <gtest/gtest.h>
#include <cmath>

#include "mesh.hpp"
#include "p2g/interpolation_method.hpp"

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

TEST(MeshInterpolation, CellAverageOneParticle)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    std::vector<double> x = {0.375};
    std::vector<double> y = {0.375};
    std::vector<double> z = {0.375};
    std::vector<double> mass = {8.0};

    mesh.rasterize_particles_to_mesh_cell_average(x, y, z, mass);

    double cellSize = 0.25;
    double cellVolume = cellSize * cellSize * cellSize;
    double expectedDensity = 8.0 / cellVolume;

    double totalDens = 0;
    for (size_t i = 0; i < mesh.dens().size(); ++i) totalDens += mesh.dens()[i] * cellVolume;
    EXPECT_NEAR(totalDens, 8.0, 1e-10);

    auto [ii, jj, kk] = mesh.positionToCell(0.375, 0.375, 0.375);
    size_t idx = ii + jj * 4 + kk * 16;
    EXPECT_NEAR(mesh.dens()[idx], expectedDensity, 1e-10);
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
    // SPH deposits density (mass*W). Kernel normalization and discrete sum may vary.
    EXPECT_GT(totalMass, 0.1);
    EXPECT_LT(totalMass, 20.0);
    // Peak density should be at/near particle
    EXPECT_GT(mesh.dens()[4 + 4*8 + 4*64], 0.0);
}

TEST(MeshInterpolation, MultiFieldCellAverageOneParticle)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    std::vector<double> x = {0.375}, y = {0.375}, z = {0.375};
    std::vector<double> mass = {8.0};
    std::vector<double> temp = {2.0};
    std::vector<std::vector<double>*> field_ptrs = {&mass, &temp};

    mesh.rasterize_particles_to_mesh_cell_average_multi(x, y, z, field_ptrs, 2);

    double cellSize   = 0.25;
    double cellVolume = cellSize * cellSize * cellSize;
    auto [ii, jj, kk] = mesh.positionToCell(0.375, 0.375, 0.375);
    size_t idx = ii + jj * 4 + kk * 16;

    EXPECT_NEAR(mesh.grid_fields_[0][idx], 8.0 / cellVolume, 1e-10);
    EXPECT_NEAR(mesh.grid_fields_[1][idx], 2.0 / cellVolume, 1e-10);
    double totalMass = 0, totalTemp = 0;
    for (size_t i = 0; i < mesh.grid_fields_[0].size(); ++i)
    {
        totalMass += mesh.grid_fields_[0][i] * cellVolume;
        totalTemp += mesh.grid_fields_[1][i] * cellVolume;
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
    EXPECT_GT(totalMass, 0.1);
    EXPECT_LT(totalMass, 20.0);
    EXPECT_GT(totalTemp, 0.05);
    EXPECT_LT(totalTemp, 10.0);
    EXPECT_NEAR(totalTemp / totalMass, 0.5, 0.1);
    EXPECT_GT(mesh.grid_fields_[0][4 + 4*8 + 4*64], 0.0);
    EXPECT_GT(mesh.grid_fields_[1][4 + 4*8 + 4*64], 0.0);
}
