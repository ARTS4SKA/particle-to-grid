// P2G -> G2P round trip against the particle data in data/turb_50.h5.
//
// Both transfers are cell-average, so the round trip is lossy by construction: a particle gets
// back the mean over all particles sharing its cell, which equals its own value only when it is
// alone in that cell. The tests below check both regimes, plus the invariants that hold either
// way (per-cell means, and the particle-sum the averaging preserves).

#include <mpi.h>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "cstone/sfc/sfc.hpp"
#include "ifile_io_impl.h"
#include "mesh.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

using namespace sphexa;

namespace {

// turb_50.h5 positions live in [-0.5, 0.5], the same box p2g::run() uses for HDF5 input.
const cstone::Box<double> testBox(-0.5, 0.5, cstone::BoundaryType::periodic);

struct Particles
{
    std::vector<double>       m, temp;
    std::vector<double>       x, y, z, h; // positions and smoothing lengths, for SPH
    std::vector<p2g::KeyType> keys;

    size_t size() const { return keys.size(); }
};

// This rank's share of the particles, with the Hilbert keys domain.sync() would assign.
bool loadTurb50(Particles& p)
{
    const std::string path = std::string(TEST_DATA_DIR) + "/turb_50.h5";
    if (!std::filesystem::exists(path)) return false;

    std::unique_ptr<IFileReader> reader = makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(path, 0, FileMode::collective);

    const size_t n = reader->localNumParticles();
    p.x.resize(n);
    p.y.resize(n);
    p.z.resize(n);
    p.h.resize(n);
    p.m.resize(n);
    p.temp.resize(n);
    reader->readField("x", p.x.data());
    reader->readField("y", p.y.data());
    reader->readField("z", p.z.data());
    reader->readField("h", p.h.data());
    reader->readField("m", p.m.data());
    reader->readField("temp", p.temp.data());
    reader->closeStep();

    p.keys.resize(n);
    for (size_t i = 0; i < n; ++i)
        p.keys[i] = cstone::sfc3D<cstone::SfcKind<p2g::KeyType>>(p.x[i], p.y[i], p.z[i], testBox);

    return true;
}

// Per-cell means over all ranks, computed without CartesianGrid, indexed by global cell
// i + j*gridDim + k*gridDim^2.
struct Reference
{
    std::vector<double> mean[2];
    std::vector<int>    counts;

    size_t cell(const p2g::KeyType key, int gridDim) const
    {
        int i, j, k;
        p2g::keyToCell(key, gridDim, i, j, k);
        return size_t(i) + size_t(j) * gridDim + size_t(k) * gridDim * gridDim;
    }
};

Reference buildReference(const Particles& p, int gridDim)
{
    const size_t nCells = size_t(gridDim) * gridDim * gridDim;
    Reference    ref;
    ref.mean[0].assign(nCells, 0.0);
    ref.mean[1].assign(nCells, 0.0);
    ref.counts.assign(nCells, 0);

    for (size_t i = 0; i < p.size(); ++i)
    {
        const size_t c = ref.cell(p.keys[i], gridDim);
        ref.mean[0][c] += p.m[i];
        ref.mean[1][c] += p.temp[i];
        ref.counts[c]++;
    }

    MPI_Allreduce(MPI_IN_PLACE, ref.mean[0].data(), nCells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ref.mean[1].data(), nCells, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ref.counts.data(), nCells, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    for (size_t c = 0; c < nCells; ++c)
        if (ref.counts[c] > 0)
        {
            ref.mean[0][c] /= ref.counts[c];
            ref.mean[1][c] /= ref.counts[c];
        }
    return ref;
}

double relError(double a, double b) { return std::abs(a - b) / std::max({std::abs(a), std::abs(b), 1e-300}); }

double globalSum(const std::vector<double>& v)
{
    double local = 0.0;
    for (double x : v)
        local += x;
    double total = 0.0;
    MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

// Runs p2g() then g2p() on mass and temperature, and returns what each particle got back.
void roundTrip(Particles& p, p2g::CartesianGrid<double>& grid, std::vector<double>& outM,
               std::vector<double>& outTemp)
{
    std::vector<std::vector<double>*> in{&p.m, &p.temp};
    grid.p2g(p.keys, in);

    outM.assign(p.size(), 0.0);
    outTemp.assign(p.size(), 0.0);
    std::vector<std::vector<double>*> out{&outM, &outTemp};
    grid.g2p(p.keys, out);
}

} // namespace

// Coarse grid: many particles per cell, so the round trip returns cell means rather than the
// original values. Checks the grid itself, the values handed back, and sum preservation.
TEST(RoundTrip, CoarseGridReturnsCellMeans)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Particles p;
    if (!loadTurb50(p)) GTEST_SKIP() << "Test data file not found: " TEST_DATA_DIR "/turb_50.h5";

    const int gridDim = 16;
    ASSERT_EQ(gridDim % numRanks, 0) << "grid must be divisible by the rank count";

    const Reference ref = buildReference(p, gridDim);
    ASSERT_GT(*std::max_element(ref.counts.begin(), ref.counts.end()), 1)
        << "expected shared cells, otherwise this test duplicates the fine-grid one";

    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim);
    std::vector<double>        outM, outTemp;
    roundTrip(p, grid, outM, outTemp);

    // 1. The grid holds the mean of the particles in each cell it owns.
    const int    slab = gridDim / numRanks;
    double       gridErr = 0.0;
    size_t       emptyNonZero = 0;
    for (int localK = 0; localK < slab; ++localK)
        for (int j = 0; j < gridDim; ++j)
            for (int i = 0; i < gridDim; ++i)
            {
                const size_t local  = size_t(i) + size_t(j) * gridDim + size_t(localK) * gridDim * gridDim;
                const size_t global = size_t(i) + size_t(j) * gridDim +
                                      size_t(rank * slab + localK) * gridDim * gridDim;
                for (int f = 0; f < 2; ++f)
                {
                    if (ref.counts[global] == 0)
                        emptyNonZero += (grid.grid_fields_[f][local] != 0.0);
                    else
                        gridErr = std::max(gridErr, relError(grid.grid_fields_[f][local], ref.mean[f][global]));
                }
            }
    EXPECT_EQ(emptyNonZero, 0u) << "cells without particles must stay zero";
    EXPECT_LT(gridErr, 1e-12) << "grid cell values differ from the reference means";

    // 2. Every particle gets its own cell's mean back.
    double g2pErr = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
    {
        const size_t c = ref.cell(p.keys[i], gridDim);
        g2pErr         = std::max({g2pErr, relError(outM[i], ref.mean[0][c]), relError(outTemp[i], ref.mean[1][c])});
    }
    EXPECT_LT(g2pErr, 1e-12) << "g2p values differ from the reference cell means";

    // 3. Averaging is mean-preserving, so the sum over all particles survives the round trip.
    EXPECT_LT(relError(globalSum(outM), globalSum(p.m)), 1e-10);
    EXPECT_LT(relError(globalSum(outTemp), globalSum(p.temp)), 1e-10);
}

// Fine grid: most particles end up alone in a cell, and those must get their exact input back.
TEST(RoundTrip, FineGridRecoversOriginalValues)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Particles p;
    if (!loadTurb50(p)) GTEST_SKIP() << "Test data file not found: " TEST_DATA_DIR "/turb_50.h5";

    const int gridDim = 128;
    ASSERT_EQ(gridDim % numRanks, 0) << "grid must be divisible by the rank count";

    const Reference ref = buildReference(p, gridDim);

    p2g::CartesianGrid<double> grid(rank, numRanks, gridDim);
    std::vector<double>        outM, outTemp;
    roundTrip(p, grid, outM, outTemp);

    size_t alone = 0, shared = 0, mismatches = 0;
    double sharedErr = 0.0;
    for (size_t i = 0; i < p.size(); ++i)
    {
        const size_t c = ref.cell(p.keys[i], gridDim);
        if (ref.counts[c] == 1)
        {
            ++alone;
            // Exact: p2g divides the single value by a count of 1, and g2p copies it back.
            if (outM[i] != p.m[i] || outTemp[i] != p.temp[i]) ++mismatches;
        }
        else
        {
            ++shared;
            sharedErr = std::max({sharedErr, relError(outM[i], ref.mean[0][c]), relError(outTemp[i], ref.mean[1][c])});
        }
    }

    EXPECT_EQ(mismatches, 0u) << "particles alone in a cell must get their original values back";
    EXPECT_LT(sharedErr, 1e-12) << "particles sharing a cell must get that cell's mean";

    uint64_t counts[2] = {alone, shared}, totals[2] = {0, 0};
    MPI_Allreduce(counts, totals, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) std::cout << "alone in cell: " << totals[0] << ", sharing: " << totals[1] << std::endl;
    EXPECT_GT(totals[0], totals[1]) << "expected most particles to be alone at gridDim=" << gridDim;
}

// With residuals the round trip is invertible regardless of how many particles share a cell.
// gridDim=4 puts ~2000 particles in every cell, the worst case for the plain cell mean.
TEST(RoundTrip, ResidualsRecoverOriginalValues)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Particles p;
    if (!loadTurb50(p)) GTEST_SKIP() << "Test data file not found: " TEST_DATA_DIR "/turb_50.h5";

    for (int gridDim : {4, 16, 128})
    {
        if (gridDim % numRanks != 0) continue;
        const Reference ref = buildReference(p, gridDim);

        p2g::CartesianGrid<double> grid(rank, numRanks, gridDim);

        std::vector<std::vector<double>*> fields = {&p.m, &p.temp};
        std::vector<double>               resM(p.size()), resTemp(p.size());
        std::vector<std::vector<double>*> residuals = {&resM, &resTemp};
        grid.p2g(p.keys, fields, residuals);

        std::vector<double>               outM(p.size(), 0.0), outTemp(p.size(), 0.0);
        std::vector<std::vector<double>*> out = {&outM, &outTemp};
        grid.g2p(p.keys, out, residuals);

        // 1. Every particle gets its own value back, however crowded its cell is.
        double maxErr   = 0.0;
        size_t bitExact = 0;
        for (size_t i = 0; i < p.size(); ++i)
        {
            maxErr = std::max({maxErr, relError(outM[i], p.m[i]), relError(outTemp[i], p.temp[i])});
            bitExact += (outM[i] == p.m[i] && outTemp[i] == p.temp[i]);
        }
        EXPECT_LT(maxErr, 1e-14) << "gridDim=" << gridDim << ": residual round trip is not exact";

        // 2. The grid is untouched by the lossless path: it still holds the cell means.
        const int slab    = gridDim / numRanks;
        double    gridErr = 0.0;
        for (int localK = 0; localK < slab; ++localK)
            for (int j = 0; j < gridDim; ++j)
                for (int i = 0; i < gridDim; ++i)
                {
                    const size_t local  = size_t(i) + size_t(j) * gridDim + size_t(localK) * gridDim * gridDim;
                    const size_t global = size_t(i) + size_t(j) * gridDim +
                                          size_t(rank * slab + localK) * gridDim * gridDim;
                    if (ref.counts[global] == 0) continue;
                    for (int f = 0; f < 2; ++f)
                        gridErr = std::max(gridErr, relError(grid.grid_fields_[f][local], ref.mean[f][global]));
                }
        EXPECT_LT(gridErr, 1e-12) << "gridDim=" << gridDim << ": grid no longer holds cell means";

        // 3. Residuals within a cell cancel, which is why the mean survives round-tripping.
        std::vector<double> resSum(size_t(gridDim) * gridDim * gridDim, 0.0);
        for (size_t i = 0; i < p.size(); ++i)
            resSum[ref.cell(p.keys[i], gridDim)] += resM[i];
        MPI_Allreduce(MPI_IN_PLACE, resSum.data(), resSum.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double maxResSum = 0.0;
        for (size_t c = 0; c < resSum.size(); ++c)
            if (ref.counts[c] > 0) maxResSum = std::max(maxResSum, std::abs(resSum[c] / ref.mean[0][c]));
        EXPECT_LT(maxResSum, 1e-10) << "gridDim=" << gridDim << ": residuals in a cell do not cancel";

        uint64_t local[2] = {bitExact, p.size()}, total[2] = {0, 0};
        MPI_Allreduce(local, total, 2, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "gridDim=" << gridDim << ": bit-exact " << total[0] << "/" << total[1]
                      << ", max rel error " << maxErr << std::endl;
    }
}

// Density reduction on real data: total mass must survive the transfer, and the field must
// actually vary (the count-average returned the same constant in every occupied cell).
TEST(Density, ConservesTotalMassOnRealData)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Particles p;
    if (!loadTurb50(p)) GTEST_SKIP() << "Test data file not found: " TEST_DATA_DIR "/turb_50.h5";

    for (int gridDim : {16, 64})
    {
        if (gridDim % numRanks != 0) continue;

        p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, -0.5, 0.5);
        grid.setReduction(0, p2g::Reduction::Density);

        std::vector<std::vector<double>*> fields = {&p.m};
        grid.p2g(p.keys, fields);

        // Mass in == mass out.
        double gridMass = 0.0, minRho = 1e300, maxRho = -1e300;
        for (double rho : grid.dens())
        {
            gridMass += rho * grid.cellVolume();
            minRho = std::min(minRho, rho);
            maxRho = std::max(maxRho, rho);
        }
        double totalGridMass = 0.0;
        MPI_Allreduce(&gridMass, &totalGridMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_LT(relError(totalGridMass, globalSum(p.m)), 1e-12)
            << "gridDim=" << gridDim << ": mass not conserved";

        // The density field must have structure, not one repeated value.
        double extremes[2] = {0, 0}, local[2] = {-minRho, maxRho};
        MPI_Allreduce(local, extremes, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        const double globalMin = -extremes[0], globalMax = extremes[1];
        EXPECT_GT(globalMax, globalMin) << "gridDim=" << gridDim << ": density field is constant";
        if (rank == 0)
            std::cout << "gridDim=" << gridDim << ": density range [" << globalMin << ", " << globalMax << "]"
                      << std::endl;
    }
}

// SPH deposition on real data, across ranks. Particles near a slab boundary spread into cells
// owned by other ranks, so this exercises the weighted exchange that nearest-cell binning skips.
TEST(Sph, ConservesMassAcrossRanksOnRealData)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    Particles p;
    if (!loadTurb50(p)) GTEST_SKIP() << "Test data file not found: " TEST_DATA_DIR "/turb_50.h5";

    const double particleMass = globalSum(p.m);
    double       hMin = 1e300, hMax = -1e300;
    for (double h : p.h)
    {
        hMin = std::min(hMin, h);
        hMax = std::max(hMax, h);
    }

    for (int gridDim : {16, 32})
    {
        if (gridDim % numRanks != 0) continue;

        p2g::CartesianGrid<double> grid(rank, numRanks, gridDim, -0.5, 0.5);
        grid.ensureNumFields(2);
        grid.setReduction(0, p2g::Reduction::Density);
        grid.setReduction(1, p2g::Reduction::Mean);

        std::vector<std::vector<double>*> fields{&p.m, &p.temp};
        grid.p2gSph(p.x, p.y, p.z, p.h, fields);

        double localMass = 0.0, localPeak = 0.0;
        size_t localOccupied = 0;
        for (double rho : grid.dens())
        {
            localMass += rho * grid.cellVolume();
            localPeak = std::max(localPeak, rho);
            localOccupied += (rho != 0.0);
        }
        double gridMass = 0.0;
        MPI_Allreduce(&localMass, &gridMass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_LT(relError(gridMass, particleMass), 1e-10)
            << "gridDim=" << gridDim << ": SPH lost mass across ranks";

        // Temperatures must stay inside the range of the inputs: a weighted average cannot
        // exceed its samples, and a wrong normalisation would show up here immediately.
        double localTempMax = 0.0, inputTempMax = 0.0;
        for (double t : grid.grid_fields_[1])
            localTempMax = std::max(localTempMax, t);
        for (double t : p.temp)
            inputTempMax = std::max(inputTempMax, t);
        double tempMax = 0.0, inputMax = 0.0;
        MPI_Allreduce(&localTempMax, &tempMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&inputTempMax, &inputMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        EXPECT_LE(tempMax, inputMax * (1 + 1e-12)) << "gridDim=" << gridDim << ": weighted mean out of range";

        // Compare against nearest-cell binning on the same data.
        p2g::CartesianGrid<double> ngp(rank, numRanks, gridDim, -0.5, 0.5);
        ngp.setReduction(0, p2g::Reduction::Density);
        std::vector<std::vector<double>*> massOnly{&p.m};
        ngp.p2g(p.keys, massOnly);

        double ngpLocalPeak = 0.0;
        size_t ngpOccupied  = 0;
        for (double rho : ngp.dens())
        {
            ngpLocalPeak = std::max(ngpLocalPeak, rho);
            ngpOccupied += (rho != 0.0);
        }
        double   sphPeak = 0.0, ngpPeak = 0.0;
        uint64_t sphCells = 0, ngpCells = 0;
        MPI_Allreduce(&localPeak, &sphPeak, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&ngpLocalPeak, &ngpPeak, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        uint64_t sphLocal = localOccupied, ngpLocal = ngpOccupied;
        MPI_Allreduce(&sphLocal, &sphCells, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&ngpLocal, &ngpCells, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

        if (rank == 0)
            std::cout << "gridDim=" << gridDim << ": h in [" << hMin << ", " << hMax << "], cell "
                      << grid.cellSize() << " | occupied cells sph=" << sphCells << " ngp=" << ngpCells
                      << " | peak density sph=" << sphPeak << " ngp=" << ngpPeak << std::endl;
    }
}
