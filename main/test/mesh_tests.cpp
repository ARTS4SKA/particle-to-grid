#include <mpi.h>
#include "gtest/gtest.h"
#include "test_utils.hpp"

TEST(CartesianGridTest, Init)
{
    int rank = 0, numRanks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int gridSize = 16;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridSize, 0.0, 1.0);

    EXPECT_EQ(grid.gridDim_, gridSize);
    EXPECT_EQ(grid.dens().size(), (size_t)(gridSize * gridSize * gridSize) / std::max(1, numRanks));
}

TEST(CartesianGridTest, SlabRank)
{
    // gridDim=4, numRanks=2 → slab=2; k=0,1 → rank 0, k=2,3 → rank 1
    EXPECT_EQ(p2g::slabRank(0, 4, 2), 0);
    EXPECT_EQ(p2g::slabRank(1, 4, 2), 0);
    EXPECT_EQ(p2g::slabRank(2, 4, 2), 1);
    EXPECT_EQ(p2g::slabRank(3, 4, 2), 1);
}

TEST(CartesianGridTest, SlabRankClamp)
{
    // gridDim=7, numRanks=3 → slab=2; k=6 would naively give rank 3 (out of bounds)
    EXPECT_EQ(p2g::slabRank(0, 7, 3), 0);
    EXPECT_EQ(p2g::slabRank(1, 7, 3), 0);
    EXPECT_EQ(p2g::slabRank(2, 7, 3), 1);
    EXPECT_EQ(p2g::slabRank(3, 7, 3), 1);
    EXPECT_EQ(p2g::slabRank(4, 7, 3), 2);
    EXPECT_EQ(p2g::slabRank(5, 7, 3), 2);
    EXPECT_EQ(p2g::slabRank(6, 7, 3), 2);
}

TEST(CartesianGridTest, SlabLocalIndex)
{
    // gridDim=4, numRanks=2: cell (1,2,3) lives on rank 1 at local plane 1
    EXPECT_EQ(p2g::slabLocalIndex(1, 2, 3, 4, 2), 1u + 2u * 4 + 1u * 16);
    EXPECT_EQ(p2g::slabLocalIndex(1, 2, 1, 4, 2), 1u + 2u * 4 + 1u * 16);
    EXPECT_EQ(p2g::slabLocalIndex(3, 3, 0, 4, 2), 3u + 3u * 4);
}

TEST(CartesianGridTest, EnsureNumFields)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int gridSize = 8;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridSize, 0.0, 1.0);
    EXPECT_EQ(grid.numFields(), 1u);

    grid.ensureNumFields(3);
    EXPECT_EQ(grid.numFields(), 3u);

    for (size_t f = 0; f < 3; ++f)
        EXPECT_EQ(grid.grid_fields_[f].size(), grid.localSize());
}

TEST(CartesianGridTest, KeyToCellRoundTrip)
{
    const int gridDim = 8;
    for (int i = 0; i < gridDim; ++i)
        for (int j = 0; j < gridDim; ++j)
            for (int k = 0; k < gridDim; ++k)
            {
                int ri, rj, rk;
                p2g::keyToCell(cellToKey(i, j, k, gridDim), gridDim, ri, rj, rk);
                EXPECT_EQ(ri, i) << "i=" << i << " j=" << j << " k=" << k;
                EXPECT_EQ(rj, j);
                EXPECT_EQ(rk, k);
            }
}
