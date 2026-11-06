// CPU/GPU agreement for SPH deposition. Built only when CUDA is available, and compiled with
// USE_CUDA so that p2gSph() dispatches to the device while p2gSphCpu() stays on the host.
//
// The two are not expected to match bit for bit: the GPU accumulates with atomics, so the
// summation order differs from the host loop. Running the CPU path on one rank versus four shows
// the same few-ulp spread, so the tolerance below is about floating-point associativity, not the
// backend.

#include <mpi.h>
#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "ifile_io_impl.h"
#include "mesh.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

using namespace sphexa;

namespace {

constexpr double LMIN = 0.0, LMAX = 1.0;
constexpr double TOLERANCE = 1e-12; // measured agreement is ~2e-15

struct Field
{
    std::vector<double> values;
    p2g::Reduction      reduction;
};

double relError(double a, double b)
{
    const double scale = std::max({std::abs(a), std::abs(b), 1e-300});
    return std::abs(a - b) / scale;
}

// Compare the host and device results for the same inputs.
void expectCpuGpuAgree(int rank, int numRanks, int gridDim, const std::vector<double>& x,
                       const std::vector<double>& y, const std::vector<double>& z,
                       const std::vector<double>& h, std::vector<Field>& fields, int maxSupportCells = 8,
                       bool periodic = true)
{
    std::vector<std::vector<double>*> ptrs;
    for (auto& f : fields)
        ptrs.push_back(&f.values);

    p2g::CartesianGrid<double> cpu(rank, numRanks, gridDim, LMIN, LMAX);
    p2g::CartesianGrid<double> gpu(rank, numRanks, gridDim, LMIN, LMAX);
    cpu.ensureNumFields(fields.size());
    gpu.ensureNumFields(fields.size());
    for (size_t f = 0; f < fields.size(); ++f)
    {
        cpu.setReduction(f, fields[f].reduction);
        gpu.setReduction(f, fields[f].reduction);
    }

    cpu.p2gSphCpu(x, y, z, h, ptrs, maxSupportCells, periodic);
    gpu.p2gSph(x, y, z, h, ptrs, maxSupportCells, periodic); // dispatches to CUDA

    double worst = 0.0;
    size_t nonZero = 0;
    for (size_t f = 0; f < fields.size(); ++f)
        for (size_t c = 0; c < cpu.localSize(); ++c)
        {
            worst = std::max(worst, relError(cpu.grid_fields_[f][c], gpu.grid_fields_[f][c]));
            nonZero += (cpu.grid_fields_[f][c] != 0.0);
        }

    EXPECT_LT(worst, TOLERANCE) << "gridDim=" << gridDim << ": CPU and GPU disagree";
    uint64_t localNonZero = nonZero, totalNonZero = 0;
    MPI_Allreduce(&localNonZero, &totalNonZero, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    EXPECT_GT(totalNonZero, 0u) << "gridDim=" << gridDim << ": nothing was deposited";

    // Mass must be conserved on both paths.
    if (fields[0].reduction == p2g::Reduction::Density)
    {
        double localParticle = 0.0;
        for (double m : fields[0].values)
            localParticle += m;
        double localCpu = 0.0, localGpu = 0.0;
        for (size_t c = 0; c < cpu.localSize(); ++c)
        {
            localCpu += cpu.grid_fields_[0][c] * cpu.cellVolume();
            localGpu += gpu.grid_fields_[0][c] * gpu.cellVolume();
        }
        double in[3] = {localParticle, localCpu, localGpu}, total[3] = {0, 0, 0};
        MPI_Allreduce(in, total, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_LT(relError(total[1], total[0]), 1e-10) << "CPU SPH lost mass";
        EXPECT_LT(relError(total[2], total[0]), 1e-10) << "GPU SPH lost mass";
    }
}

// Deterministic pseudo-random particles, identical on every rank for a given seed.
void makeParticles(size_t n, unsigned seed, std::vector<double>& x, std::vector<double>& y,
                   std::vector<double>& z, std::vector<double>& h, double hScale)
{
    x.resize(n);
    y.resize(n);
    z.resize(n);
    h.resize(n);
    uint64_t state = 88172645463325252ull + seed;
    auto     next  = [&state]
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return static_cast<double>(state % 1000000) / 1000000.0;
    };
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = LMIN + next() * (LMAX - LMIN);
        y[i] = LMIN + next() * (LMAX - LMIN);
        z[i] = LMIN + next() * (LMAX - LMIN);
        h[i] = hScale * (0.5 + next());
    }
}

} // namespace

TEST(SphCuda, MatchesCpuForMultiCellSupport)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int gridDim = 16;
    if (gridDim % numRanks != 0) GTEST_SKIP() << "grid not divisible by rank count";

    std::vector<double> x, y, z, h;
    makeParticles(4000, 1, x, y, z, h, 2.0 / gridDim); // support spans a few cells
    std::vector<Field> fields{{std::vector<double>(x.size(), 1.0 / x.size()), p2g::Reduction::Density}};

    expectCpuGpuAgree(rank, numRanks, gridDim, x, y, z, h, fields);
}

TEST(SphCuda, MatchesCpuForSubCellSupport)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int gridDim = 16;
    if (gridDim % numRanks != 0) GTEST_SKIP() << "grid not divisible by rank count";

    // h far below the cell size, so both paths must take the single-cell fallback.
    std::vector<double> x, y, z, h;
    makeParticles(2000, 2, x, y, z, h, 1e-7 / gridDim);
    std::vector<Field> fields{{std::vector<double>(x.size(), 3.0), p2g::Reduction::Density}};

    expectCpuGpuAgree(rank, numRanks, gridDim, x, y, z, h, fields);
}

TEST(SphCuda, MatchesCpuForMixedReductionsAndWideSupport)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int gridDim = 16;
    if (gridDim % numRanks != 0) GTEST_SKIP() << "grid not divisible by rank count";

    std::vector<double> x, y, z, h;
    makeParticles(1500, 3, x, y, z, h, 4.0 / gridDim); // wide support, crosses slab boundaries

    std::vector<double> mass(x.size(), 2.0 / x.size()), temp(x.size());
    for (size_t i = 0; i < temp.size(); ++i)
        temp[i] = 100.0 + 50.0 * std::sin(0.3 * i);

    std::vector<Field> fields{{mass, p2g::Reduction::Density}, {temp, p2g::Reduction::Mean}};
    expectCpuGpuAgree(rank, numRanks, gridDim, x, y, z, h, fields);
}

TEST(SphCuda, MatchesCpuForOpenBoundaries)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int gridDim = 16;
    if (gridDim % numRanks != 0) GTEST_SKIP() << "grid not divisible by rank count";

    std::vector<double> x, y, z, h;
    makeParticles(1500, 4, x, y, z, h, 3.0 / gridDim);
    std::vector<Field> fields{{std::vector<double>(x.size(), 1.0), p2g::Reduction::Density}};

    expectCpuGpuAgree(rank, numRanks, gridDim, x, y, z, h, fields, 8, /*periodic=*/false);
}

TEST(SphCuda, MatchesCpuOnTurb50)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const std::string path = std::string(TEST_DATA_DIR) + "/turb_50.h5";
    if (!std::filesystem::exists(path)) GTEST_SKIP() << "Test data file not found: " << path;

    std::unique_ptr<IFileReader> reader = makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(path, 0, FileMode::collective);
    const size_t        n = reader->localNumParticles();
    std::vector<double> x(n), y(n), z(n), h(n), m(n), temp(n);
    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());
    reader->readField("h", h.data());
    reader->readField("m", m.data());
    reader->readField("temp", temp.data());
    reader->closeStep();

    // turb_50 lives in [-0.5, 0.5]; shift it into the unit box these helpers assume.
    for (size_t i = 0; i < n; ++i)
    {
        x[i] += 0.5;
        y[i] += 0.5;
        z[i] += 0.5;
    }

    for (int gridDim : {16, 32})
    {
        if (gridDim % numRanks != 0) continue;
        std::vector<Field> fields{{m, p2g::Reduction::Density}, {temp, p2g::Reduction::Mean}};
        expectCpuGpuAgree(rank, numRanks, gridDim, x, y, z, h, fields);
    }
}
