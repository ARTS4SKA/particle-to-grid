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
    EXPECT_EQ(mesh.dens_.size(), gridSize3 / std::max(1, numRanks));
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
