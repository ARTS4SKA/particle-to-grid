#include <mpi.h>
#include "gtest/gtest.h"
#include "mesh.hpp"

TEST(meshTest, testMeshInit)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int                    gridSize  = 16;
    int                    gridSize3 = gridSize * gridSize * gridSize;
    p2g::Mesh<double>      mesh(rank, numRanks, gridSize, 0.0, 1.0);

    EXPECT_EQ(mesh.gridDim_, gridSize);
    EXPECT_EQ(mesh.dens().size(), gridSize3 / std::max(1, numRanks));
}

TEST(meshTest, testCalculateRankFromMeshCoord)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int               gridSize  = 4;
    int               nRanks    = 2;
    p2g::Mesh<double> mesh(0, nRanks, gridSize, 0.0, 1.0);

    // Rank from z-index k: rank = k / (gridDim / numRanks), base = 2, so k=0,1 -> 0, k=2,3 -> 1
    std::vector<int> solution = {0, 0, 1, 1};
    for (int k = 0; k < gridSize; ++k)
    {
        EXPECT_EQ(mesh.calculateRankFromMeshCoord(k), solution[k]);
    }
}

TEST(meshTest, testEnsureNumFields)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int               gridSize = 8;
    p2g::Mesh<double> mesh(rank, numRanks, gridSize, 0.0, 1.0);
    EXPECT_EQ(mesh.numFields(), 1u);
    EXPECT_EQ(mesh.grid_fields_.size(), 1u);

    mesh.ensureNumFields(3);
    EXPECT_EQ(mesh.numFields(), 3u);
    EXPECT_EQ(mesh.grid_fields_.size(), 3u);
    size_t localSize = static_cast<size_t>(gridSize) * gridSize * gridSize;
    for (size_t f = 0; f < 3; ++f)
    {
        EXPECT_EQ(mesh.grid_fields_[f].size(), localSize);
    }
}

TEST(meshTest, testSetOutputFieldIndex)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    mesh.ensureNumFields(2);
    // Cell (0,0,0) is always on rank 0 (local index 0)
    mesh.setOutputFieldIndex(0);
    mesh.assignDensityByMeshCoord(0, 0, 0, 5.0);
    mesh.setOutputFieldIndex(1);
    mesh.assignDensityByMeshCoord(0, 0, 0, 10.0);

    if (rank == 0)
    {
        EXPECT_DOUBLE_EQ(mesh.grid_fields_[0][0], 5.0);
        EXPECT_DOUBLE_EQ(mesh.grid_fields_[1][0], 10.0);
    }
}

TEST(meshTest, testAssignValuesByMeshCoordAndSingleExchange)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    mesh.ensureNumFields(2);
    mesh.resetCommAndDens();
    double vals[2] = {8.0, 4.0};
    mesh.assignValuesByMeshCoord(0, 0, 0, vals, 2);
    mesh.performExchangeAndAccumulate(2);
    mesh.convertMassToDensityAllFields(2);

    double cellVolume = 0.25 * 0.25 * 0.25;
    if (rank == 0)
    {
        EXPECT_NEAR(mesh.grid_fields_[0][0], 8.0 / cellVolume, 1e-10);
        EXPECT_NEAR(mesh.grid_fields_[1][0], 4.0 / cellVolume, 1e-10);
    }
}

TEST(meshTest, testRasterizeMultiCellAverageTwoFields)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::Mesh<double> mesh(rank, numRanks, 4, 0.0, 1.0);
    mesh.ensureNumFields(2);
    // One particle at (0.125, 0.125, 0.125) -> cell (0,0,0), on rank 0
    std::vector<double> x = {0.125}, y = {0.125}, z = {0.125};
    std::vector<double> mass = {6.0};
    std::vector<double> temp = {3.0};
    std::vector<std::vector<double>*> field_ptrs = {&mass, &temp};

    mesh.rasterize_particles_to_mesh_cell_average_multi(x, y, z, field_ptrs, 2);

    double cellVolume = 0.25 * 0.25 * 0.25;
    if (rank == 0)
    {
        EXPECT_NEAR(mesh.grid_fields_[0][0], 6.0 / cellVolume, 1e-10);
        EXPECT_NEAR(mesh.grid_fields_[1][0], 3.0 / cellVolume, 1e-10);
    }
}
