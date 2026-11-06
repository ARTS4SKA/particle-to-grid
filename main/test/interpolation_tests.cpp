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

// -----------------------------------------------------------------------
// Invertible P2G/G2P — cell mean plus per-particle residual
// -----------------------------------------------------------------------

TEST(Residuals, SharedCellRoundTripIsExact)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    // Three particles in one cell, one in another.
    std::vector<p2g::KeyType> keys = {cellToKey(0, 0, 0, 4), cellToKey(0, 0, 0, 4), cellToKey(0, 0, 0, 4),
                                      cellToKey(2, 2, 2, 4)};
    std::vector<double>       mass = {1.0, 2.0, 6.0, 7.0};
    std::vector<double>       temp = {10.0, 20.0, 30.0, 40.0};
    const std::vector<double> origMass = mass, origTemp = temp;

    std::vector<std::vector<double>*> fields = {&mass, &temp};
    std::vector<double>               resMass(keys.size()), resTemp(keys.size());
    std::vector<std::vector<double>*> residuals = {&resMass, &resTemp};

    grid.p2g(keys, fields, residuals);

    // The grid still holds plain cell means: (1+2+6)/3 = 3 and (10+20+30)/3 = 20.
    EXPECT_DOUBLE_EQ(grid.grid_fields_[0][0], 3.0);
    EXPECT_DOUBLE_EQ(grid.grid_fields_[1][0], 20.0);

    // Residuals are the deviations from that mean, and sum to zero within a cell.
    EXPECT_DOUBLE_EQ(resMass[0], -2.0);
    EXPECT_DOUBLE_EQ(resMass[1], -1.0);
    EXPECT_DOUBLE_EQ(resMass[2], 3.0);
    EXPECT_DOUBLE_EQ(resMass[0] + resMass[1] + resMass[2], 0.0);
    // The lone particle needs no correction.
    EXPECT_DOUBLE_EQ(resMass[3], 0.0);
    EXPECT_DOUBLE_EQ(resTemp[3], 0.0);

    std::vector<double>               outMass(keys.size(), 0.0), outTemp(keys.size(), 0.0);
    std::vector<std::vector<double>*> out = {&outMass, &outTemp};
    grid.g2p(keys, out, residuals);

    for (size_t p = 0; p < keys.size(); ++p)
    {
        EXPECT_DOUBLE_EQ(outMass[p], origMass[p]) << "particle " << p;
        EXPECT_DOUBLE_EQ(outTemp[p], origTemp[p]) << "particle " << p;
    }
}

TEST(Residuals, WithoutResidualsG2PStillReturnsCellMean)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    p2g::CartesianGrid<double> grid(rank, numRanks, 4, 0.0, 1.0);

    std::vector<p2g::KeyType>         keys = {cellToKey(1, 1, 1, 4), cellToKey(1, 1, 1, 4)};
    std::vector<double>               mass = {4.0, 8.0};
    std::vector<std::vector<double>*> fields = {&mass};
    std::vector<double>               res(keys.size());
    std::vector<std::vector<double>*> residuals = {&res};

    grid.p2g(keys, fields, residuals);

    // Ignoring the residuals must reproduce the old lossy behaviour: both get the mean.
    std::vector<double>               out(keys.size(), 0.0);
    std::vector<std::vector<double>*> outPtrs = {&out};
    grid.g2p(keys, outPtrs);

    EXPECT_DOUBLE_EQ(out[0], 6.0);
    EXPECT_DOUBLE_EQ(out[1], 6.0);
}

// -----------------------------------------------------------------------
// Density reduction — mass is summed and divided by the cell volume
// -----------------------------------------------------------------------

TEST(Density, SumsMassAndDividesByCellVolume)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 4;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, 0.0, 1.0);
    grid.setReduction(0, p2g::Reduction::Density);

    // Cell volume for a unit box at gridDim=4 is (1/4)^3 = 1/64.
    EXPECT_DOUBLE_EQ(grid.cellVolume(), 1.0 / 64.0);

    // Three particles of mass 2 in one cell, one of mass 5 in another.
    std::vector<p2g::KeyType> keys = {cellToKey(0, 0, 0, gridDim), cellToKey(0, 0, 0, gridDim),
                                      cellToKey(0, 0, 0, gridDim), cellToKey(2, 2, 2, gridDim)};
    std::vector<double>               mass   = {2.0, 2.0, 2.0, 5.0};
    std::vector<std::vector<double>*> fields = {&mass};

    grid.p2g(keys, fields);

    // Summed, not averaged: 6 / (1/64) = 384, and 5 / (1/64) = 320.
    EXPECT_DOUBLE_EQ(grid.dens()[0], 6.0 * 64.0);
    EXPECT_DOUBLE_EQ(grid.dens()[2 + 2 * gridDim + 2 * gridDim * gridDim], 5.0 * 64.0);

    // Total mass is conserved and empty cells stay empty.
    double total = 0.0, occupied = 0;
    for (double rho : grid.dens())
    {
        total += rho * grid.cellVolume();
        occupied += (rho != 0.0);
    }
    EXPECT_DOUBLE_EQ(total, 11.0);
    EXPECT_EQ(occupied, 2);
}

TEST(Density, EqualMassParticlesGiveVaryingDensity)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 4;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, 0.0, 1.0);
    grid.setReduction(0, p2g::Reduction::Density);

    // Two particles in one cell, one in another: identical masses, different densities.
    // This is the case the old count-average got wrong, returning the same value for both.
    std::vector<p2g::KeyType>         keys   = {cellToKey(0, 0, 0, gridDim), cellToKey(0, 0, 0, gridDim),
                                                cellToKey(1, 0, 0, gridDim)};
    std::vector<double>               mass   = {3.0, 3.0, 3.0};
    std::vector<std::vector<double>*> fields = {&mass};

    grid.p2g(keys, fields);

    EXPECT_DOUBLE_EQ(grid.dens()[0], 2.0 * grid.dens()[1]);
}

TEST(Density, ExtraFieldsStillAveraged)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 4;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, 0.0, 1.0);
    grid.ensureNumFields(2);
    grid.setReduction(0, p2g::Reduction::Density);

    std::vector<p2g::KeyType>         keys   = {cellToKey(0, 0, 0, gridDim), cellToKey(0, 0, 0, gridDim)};
    std::vector<double>               mass   = {1.0, 3.0};
    std::vector<double>               temp   = {10.0, 20.0};
    std::vector<std::vector<double>*> fields = {&mass, &temp};

    grid.p2g(keys, fields);

    EXPECT_DOUBLE_EQ(grid.grid_fields_[0][0], 4.0 * 64.0); // mass summed -> density
    EXPECT_DOUBLE_EQ(grid.grid_fields_[1][0], 15.0);       // temperature averaged
}

// -----------------------------------------------------------------------
// SPH interpolation — kernel-weighted deposition over the support radius
// -----------------------------------------------------------------------

TEST(SphKernel, IsNormalisedOverSpace)
{
    // The M4 spline must integrate to 1 over its 2h support, otherwise deposited mass is wrong.
    // Earlier revisions of this code used a prefactor of 8/pi with q = r/h, which integrates to 8.
    const double h = 0.7;
    const double dr = h * 2.0 / 20000;
    double integral = 0.0;
    for (int n = 0; n < 20000; ++n)
    {
        const double r = (n + 0.5) * dr;
        integral += 4.0 * M_PI * r * r * p2g::sphKernel(r, h) * dr;
    }
    EXPECT_NEAR(integral, 1.0, 1e-6);

    EXPECT_GT(p2g::sphKernel(0.0, h), 0.0);
    EXPECT_EQ(p2g::sphKernel(2.0 * h, h), 0.0) << "kernel must vanish at and beyond the support";
    EXPECT_EQ(p2g::sphKernel(3.0 * h, h), 0.0);
    EXPECT_EQ(p2g::sphKernel(0.1, 0.0), 0.0) << "a non-positive h has no support";
    // Monotonically decreasing out to the support radius.
    EXPECT_GT(p2g::sphKernel(0.1 * h, h), p2g::sphKernel(0.9 * h, h));
    EXPECT_GT(p2g::sphKernel(0.9 * h, h), p2g::sphKernel(1.9 * h, h));
}

TEST(Sph, ConservesMassWhenSpreadOverManyCells)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 16;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, 0.0, 1.0);
    grid.setReduction(0, p2g::Reduction::Density);

    // h spans several cells, so the particle is spread rather than binned.
    std::vector<double>               xs{0.5}, ys{0.5}, zs{0.5}, hs{3.0 / gridDim}, mass{7.0};
    std::vector<std::vector<double>*> fields{&mass};

    grid.p2gSph(xs, ys, zs, hs, fields);

    double total = 0.0;
    size_t touched = 0;
    for (double rho : grid.dens())
    {
        total += rho * grid.cellVolume();
        touched += (rho != 0.0);
    }
    EXPECT_NEAR(total, 7.0, 1e-12) << "SPH deposition must conserve mass";
    EXPECT_GT(touched, 1u) << "the particle should reach more than one cell";
}

TEST(Sph, SubCellSmoothingLengthFallsBackToOneCell)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 8;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, 0.0, 1.0);
    grid.setReduction(0, p2g::Reduction::Density);

    // h is far below the cell size, so no cell center lies inside the support. The mass must
    // still land somewhere: in the cell containing the particle. This is the common case for
    // cosmological runs, where h inside a halo is tiny compared with a cell.
    const double cellSize = 1.0 / gridDim;
    std::vector<double>               xs{0.5 + 0.1 * cellSize}, ys{0.5}, zs{0.5};
    std::vector<double>               hs{1e-6 * cellSize}, mass{3.0};
    std::vector<std::vector<double>*> fields{&mass};

    grid.p2gSph(xs, ys, zs, hs, fields);

    double total = 0.0;
    size_t touched = 0;
    for (double rho : grid.dens())
    {
        total += rho * grid.cellVolume();
        touched += (rho != 0.0);
    }
    EXPECT_NEAR(total, 3.0, 1e-12) << "mass must not vanish when h is below the cell size";
    EXPECT_EQ(touched, 1u) << "all of it belongs to the particle's own cell";

    // And it is the cell the particle sits in.
    const int i = grid.positionToCell(xs[0]), j = grid.positionToCell(ys[0]), k = grid.positionToCell(zs[0]);
    EXPECT_GT(grid.dens()[p2g::slabLocalIndex(i, j, k, gridDim, numRanks)], 0.0);
}

TEST(Sph, WrapsAcrossThePeriodicBoundary)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 16;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, 0.0, 1.0);
    grid.setReduction(0, p2g::Reduction::Density);

    // A particle in the corner spills over every boundary; with wrapping none of it is lost.
    std::vector<double>               xs{0.001}, ys{0.001}, zs{0.001}, hs{2.5 / gridDim}, mass{5.0};
    std::vector<std::vector<double>*> fields{&mass};

    grid.p2gSph(xs, ys, zs, hs, fields, 8, /*periodic=*/true);

    double periodicTotal = 0.0;
    for (double rho : grid.dens())
        periodicTotal += rho * grid.cellVolume();
    EXPECT_NEAR(periodicTotal, 5.0, 1e-12);

    // The far corner must receive something, which only happens if the stencil wrapped.
    EXPECT_GT(grid.dens()[p2g::slabLocalIndex(gridDim - 1, gridDim - 1, gridDim - 1, gridDim, numRanks)], 0.0);

    // Without wrapping the weights are renormalised over the cells inside the box, so mass is
    // still conserved, but nothing reaches the opposite corner.
    grid.p2gSph(xs, ys, zs, hs, fields, 8, /*periodic=*/false);
    double openTotal = 0.0;
    for (double rho : grid.dens())
        openTotal += rho * grid.cellVolume();
    EXPECT_NEAR(openTotal, 5.0, 1e-12);
    EXPECT_DOUBLE_EQ(grid.dens()[p2g::slabLocalIndex(gridDim - 1, gridDim - 1, gridDim - 1, gridDim, numRanks)], 0.0);
}

TEST(Sph, MeanFieldIsKernelWeightedAverage)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 8;
    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, 0.0, 1.0);
    grid.ensureNumFields(2);
    grid.setReduction(0, p2g::Reduction::Density);
    grid.setReduction(1, p2g::Reduction::Mean);

    // Two particles at the same place with the same h contribute equal weights everywhere, so
    // the weighted average of their temperatures is the plain average.
    std::vector<double>               xs{0.5, 0.5}, ys{0.5, 0.5}, zs{0.5, 0.5};
    std::vector<double>               hs{2.0 / gridDim, 2.0 / gridDim};
    std::vector<double>               mass{1.0, 1.0}, temp{10.0, 30.0};
    std::vector<std::vector<double>*> fields{&mass, &temp};

    grid.p2gSph(xs, ys, zs, hs, fields);

    const int    i = grid.positionToCell(0.5);
    const size_t c = p2g::slabLocalIndex(i, i, i, gridDim, numRanks);
    EXPECT_NEAR(grid.grid_fields_[1][c], 20.0, 1e-12);

    // A Mean field must not be scaled by the cell volume the way a Density field is.
    for (double t : grid.grid_fields_[1])
        EXPECT_LE(t, 30.0 + 1e-12);
}

TEST(Sph, SmoothsMoreThanNearestCell)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 1) GTEST_SKIP() << "single-rank test";

    const int gridDim = 16;
    const int n = 400;
    std::vector<double> xs(n), ys(n), zs(n), hs(n, 2.0 / gridDim), mass(n, 1.0 / n);
    std::vector<p2g::KeyType> keys(n);
    for (int p = 0; p < n; ++p)
    {
        // Clustered in one octant, so the two schemes differ visibly.
        xs[p] = 0.25 + 0.15 * std::sin(0.7 * p);
        ys[p] = 0.25 + 0.15 * std::cos(1.1 * p);
        zs[p] = 0.25 + 0.15 * std::sin(1.9 * p);
        keys[p] = cellToKey(grid_cell(xs[p], gridDim), grid_cell(ys[p], gridDim), grid_cell(zs[p], gridDim), gridDim);
    }
    std::vector<std::vector<double>*> fields{&mass};

    p2g::CartesianGrid<double> ngp(rank, numRanks, gridDim, 0.0, 1.0);
    ngp.setReduction(0, p2g::Reduction::Density);
    ngp.p2g(keys, fields);

    p2g::CartesianGrid<double> sph(rank, numRanks, gridDim, 0.0, 1.0);
    sph.setReduction(0, p2g::Reduction::Density);
    sph.p2gSph(xs, ys, zs, hs, fields);

    double ngpMass = 0.0, sphMass = 0.0, ngpPeak = 0.0, sphPeak = 0.0;
    size_t ngpCells = 0, sphCells = 0;
    for (size_t c = 0; c < ngp.dens().size(); ++c)
    {
        ngpMass += ngp.dens()[c] * ngp.cellVolume();
        sphMass += sph.dens()[c] * sph.cellVolume();
        ngpPeak = std::max(ngpPeak, ngp.dens()[c]);
        sphPeak = std::max(sphPeak, sph.dens()[c]);
        ngpCells += (ngp.dens()[c] != 0.0);
        sphCells += (sph.dens()[c] != 0.0);
    }

    // Both conserve mass, but SPH spreads it over more cells with a lower peak.
    EXPECT_NEAR(ngpMass, 1.0, 1e-12);
    EXPECT_NEAR(sphMass, 1.0, 1e-12);
    EXPECT_GT(sphCells, ngpCells) << "SPH should occupy more cells than nearest-cell binning";
    EXPECT_LT(sphPeak, ngpPeak) << "SPH should smooth the peak down";
}
