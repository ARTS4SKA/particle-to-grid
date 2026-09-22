#include <mpi.h>
#include <gtest/gtest.h>
#include <cmath>

#include "test_utils.hpp"

// -----------------------------------------------------------------------
// P2G — cell-average
// -----------------------------------------------------------------------

TEST(P2G, OneParticleOneField)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    std::vector<p2g::KeyType> keys = { cellToKey(1, 1, 1, 4) };
    std::vector<double> mass = {8.0};
    std::vector<std::vector<double>*> field_ptrs = {&mass};

    grid.p2g(keys, field_ptrs);

    // Single particle: cell-average == particle value
    const size_t idx = 1 + 1 * 4 + 1 * 16;
    EXPECT_NEAR(grid.dens()[idx], 8.0, 1e-10);

    for (size_t i = 0; i < grid.dens().size(); ++i)
        if (i != idx) EXPECT_DOUBLE_EQ(grid.dens()[i], 0.0);
}

TEST(P2G, TwoParticlesSameCellAverages)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    std::vector<p2g::KeyType> keys = { cellToKey(0, 0, 0, 4), cellToKey(0, 0, 0, 4) };
    std::vector<double> mass = {4.0, 8.0};
    std::vector<std::vector<double>*> field_ptrs = {&mass};

    grid.p2g(keys, field_ptrs);

    // Two particles in the same cell: average = (4 + 8) / 2 = 6
    EXPECT_NEAR(grid.dens()[0], 6.0, 1e-10);
}

TEST(P2G, MultiFieldOneParticle)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    std::vector<p2g::KeyType> keys = { cellToKey(1, 1, 1, 4) };
    std::vector<double> mass = {8.0};
    std::vector<double> temp = {2.0};
    std::vector<std::vector<double>*> field_ptrs = {&mass, &temp};

    grid.p2g(keys, field_ptrs);

    const size_t idx = 1 + 1 * 4 + 1 * 16;
    EXPECT_NEAR(grid.grid_fields_[0][idx], 8.0, 1e-10);
    EXPECT_NEAR(grid.grid_fields_[1][idx], 2.0, 1e-10);

    // Only one cell non-zero
    double totalMass = 0, totalTemp = 0;
    for (size_t i = 0; i < grid.grid_fields_[0].size(); ++i)
    {
        totalMass += grid.grid_fields_[0][i];
        totalTemp += grid.grid_fields_[1][i];
    }
    EXPECT_NEAR(totalMass, 8.0, 1e-10);
    EXPECT_NEAR(totalTemp, 2.0, 1e-10);
}

// -----------------------------------------------------------------------
// G2P — cell-average reverse
// -----------------------------------------------------------------------

TEST(G2P, RoundTripSingleField)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    // Two particles in two different cells
    std::vector<p2g::KeyType> keys = { cellToKey(0, 0, 0, 4), cellToKey(1, 1, 1, 4) };
    std::vector<double> mass = {6.0, 3.0};
    std::vector<std::vector<double>*> p2g_ptrs = {&mass};

    grid.p2g(keys, p2g_ptrs);

    // G2P: each particle should read back its own cell average
    std::vector<double> particle_mass(2, 0.0);
    std::vector<std::vector<double>*> g2p_ptrs = {&particle_mass};

    grid.g2p(keys, g2p_ptrs);

    EXPECT_NEAR(particle_mass[0], 6.0, 1e-10);
    EXPECT_NEAR(particle_mass[1], 3.0, 1e-10);
}

TEST(G2P, RoundTripMultiField)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    std::vector<p2g::KeyType> keys = { cellToKey(0, 0, 0, 4), cellToKey(2, 2, 2, 4) };
    std::vector<double> mass = {10.0, 4.0};
    std::vector<double> temp = {300.0, 100.0};
    std::vector<std::vector<double>*> p2g_ptrs = {&mass, &temp};

    grid.p2g(keys, p2g_ptrs);

    std::vector<double> out_mass(2, 0.0), out_temp(2, 0.0);
    std::vector<std::vector<double>*> g2p_ptrs = {&out_mass, &out_temp};

    grid.g2p(keys, g2p_ptrs);

    EXPECT_NEAR(out_mass[0], 10.0,  1e-10);
    EXPECT_NEAR(out_mass[1],  4.0,  1e-10);
    EXPECT_NEAR(out_temp[0], 300.0, 1e-10);
    EXPECT_NEAR(out_temp[1], 100.0, 1e-10);
}

TEST(G2P, SharedCellBothParticlesGetAverage)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    // Two particles in the same cell: P2G average = (4+8)/2 = 6
    // G2P: both particles should receive 6
    std::vector<p2g::KeyType> keys = { cellToKey(0, 0, 0, 4), cellToKey(0, 0, 0, 4) };
    std::vector<double> mass = {4.0, 8.0};
    std::vector<std::vector<double>*> p2g_ptrs = {&mass};

    grid.p2g(keys, p2g_ptrs);

    std::vector<double> out_mass(2, 0.0);
    std::vector<std::vector<double>*> g2p_ptrs = {&out_mass};

    grid.g2p(keys, g2p_ptrs);

    EXPECT_NEAR(out_mass[0], 6.0, 1e-10);
    EXPECT_NEAR(out_mass[1], 6.0, 1e-10);
}
