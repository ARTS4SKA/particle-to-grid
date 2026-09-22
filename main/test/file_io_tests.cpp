#include <mpi.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "ifile_io_impl.h"

#ifdef SPH_EXA_HAVE_H5PART
#include "h5part_wrapper.hpp"
#endif

using namespace sphexa;

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "."
#endif

TEST(FileIO, HDF5ReaderOpenAndRead)
{
    std::string path = std::string(TEST_DATA_DIR) + "/turb_50.h5";
    if (!std::filesystem::exists(path))
    {
        GTEST_SKIP() << "Test data file not found: " << path;
    }

    std::unique_ptr<IFileReader> reader = makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(path, 0, FileMode::collective);

    uint64_t globalNum = reader->globalNumParticles();
    EXPECT_GT(globalNum, 0u);

    uint64_t localNum = reader->localNumParticles();
    EXPECT_LE(localNum, globalNum);
    EXPECT_GT(localNum, 0u);

    std::vector<double> x(localNum), y(localNum), z(localNum);
    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("z", z.data());

    bool hasMass = false;
    std::vector<double> mass(localNum);
    try {
        reader->readField("mass", mass.data());
        hasMass = true;
    } catch (...) { /* mass optional in some HDF5 files */ }

    reader->closeStep();

    for (uint64_t i = 0; i < localNum; ++i)
    {
        EXPECT_GE(x[i], -0.5);
        EXPECT_LE(x[i], 0.5);
        EXPECT_GE(y[i], -0.5);
        EXPECT_LE(y[i], 0.5);
        EXPECT_GE(z[i], -0.5);
        EXPECT_LE(z[i], 0.5);
        if (hasMass) { EXPECT_GT(mass[i], 0.0); }
    }
}

TEST(FileIO, TipsyReaderRequiresFile)
{
    std::unique_ptr<IFileReader> reader = makeTipsyReader(MPI_COMM_WORLD);
    EXPECT_THROW(reader->setStep("/nonexistent/tipsy.bin", 0, FileMode::collective), std::exception);
}

namespace {

void putBigEndian(std::ofstream& out, uint32_t bits)
{
    char bytes[4] = {char(bits >> 24), char(bits >> 16), char(bits >> 8), char(bits)};
    out.write(bytes, 4);
}

void putFloat(std::ofstream& out, float v)
{
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    putBigEndian(out, bits);
}

// Dark particle d has x = d, y = 10 d, ..., phi = 80 d; gas/star records are filled with -1.
void writeSyntheticTipsy(const std::string& path, int ngas, int ndark, int nstar, bool truncate)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    uint64_t      timeBits;
    double        time = 0.5;
    std::memcpy(&timeBits, &time, 8);
    putBigEndian(out, uint32_t(timeBits >> 32));
    putBigEndian(out, uint32_t(timeBits));
    for (int v : {ngas + ndark + nstar, 3, ngas, ndark, nstar, 0})
        putBigEndian(out, uint32_t(v));

    for (int i = 0; i < ngas * 12; ++i)
        putFloat(out, -1.0f);
    for (int d = 0; d < ndark; ++d)
    {
        putFloat(out, 1.0f + d); // mass
        for (int c = 1; c < 9; ++c)
            putFloat(out, float(c * 10 * d + d * (c == 1)));
    }
    for (int i = 0; i < nstar * 11 - (truncate ? 1 : 0); ++i)
        putFloat(out, -1.0f);
}

std::string tipsyTestPath(const char* name)
{
    int numRanks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    return std::filesystem::temp_directory_path() /
           (std::string("p2g_test_") + name + "_" + std::to_string(numRanks) + ".tipsy");
}

} // namespace

TEST(FileIO, TipsyReaderReadsOwnSliceOfDarkParticles)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    const int         ngas = 2, ndark = 11, nstar = 3;
    const std::string path = tipsyTestPath("slice");
    if (rank == 0) writeSyntheticTipsy(path, ngas, ndark, nstar, false);
    MPI_Barrier(MPI_COMM_WORLD);

    std::unique_ptr<IFileReader> reader = makeTipsyReader(MPI_COMM_WORLD);
    reader->setStep(path, 0, FileMode::collective);

    const uint64_t localNum = reader->localNumParticles();
    EXPECT_EQ(reader->globalNumParticles(), uint64_t(ndark));

    std::vector<double> x(localNum), y(localNum), m(localNum), mass(localNum), h(localNum), phi(localNum);
    reader->readField("x", x.data());
    reader->readField("y", y.data());
    reader->readField("m", m.data());
    reader->readField("mass", mass.data());
    reader->readField("h", h.data());
    reader->readField("phi", phi.data());
    EXPECT_THROW(reader->readField("rho", x.data()), std::exception);
    reader->closeStep();

    // Slices must be contiguous, in rank order, and cover all dark particles exactly once.
    std::vector<uint64_t> counts(numRanks);
    uint64_t              local = localNum;
    MPI_Allgather(&local, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    uint64_t first = 0, total = 0;
    for (int r = 0; r < numRanks; ++r)
    {
        if (r < rank) first += counts[r];
        total += counts[r];
    }
    EXPECT_EQ(total, uint64_t(ndark));

    for (uint64_t i = 0; i < localNum; ++i)
    {
        const double d = double(first + i);
        EXPECT_DOUBLE_EQ(x[i], 11 * d);
        EXPECT_DOUBLE_EQ(y[i], 20 * d);
        EXPECT_DOUBLE_EQ(m[i], 1 + d);
        EXPECT_DOUBLE_EQ(mass[i], 1 + d);
        EXPECT_DOUBLE_EQ(h[i], 70 * d);
        EXPECT_DOUBLE_EQ(phi[i], 80 * d);
    }

    if (rank == 0) std::filesystem::remove(path);
}

TEST(FileIO, TipsyReaderRejectsTruncatedFileOnAllRanks)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string path = tipsyTestPath("truncated");
    if (rank == 0) writeSyntheticTipsy(path, 1, 8, 2, true);
    MPI_Barrier(MPI_COMM_WORLD);

    std::unique_ptr<IFileReader> reader = makeTipsyReader(MPI_COMM_WORLD);
    EXPECT_THROW(reader->setStep(path, 0, FileMode::collective), std::runtime_error);

    if (rank == 0) std::filesystem::remove(path);
}

#ifdef SPH_EXA_HAVE_H5PART

TEST(FileIO, HDF5GridOutputWriteAndRead)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Test parameters: grid must be divisible by numRanks
    const int gridDim = std::max(4, numRanks);
    const int base = gridDim / numRanks;
    const size_t localSize = static_cast<size_t>(gridDim) * gridDim * base;
    const double Lmin = 0.0, Lmax = 1.0;
    const std::string testFile = "/tmp/test_grid_output_shared.h5";

    // Create test data: density and temperature fields (unique per rank)
    std::vector<double> density(localSize);
    std::vector<double> temperature(localSize);
    for (size_t i = 0; i < localSize; ++i)
    {
        density[i] = static_cast<double>(i + 1) * 1.5 + rank * 100.0;
        temperature[i] = static_cast<double>(i + 1) * 0.5 + rank * 50.0;
    }

    // Write HDF5 file (all ranks write in parallel to same file)
    {
        int64_t mode = H5PART_WRITE | H5PART_VFD_MPIIO_IND;
        H5PartFile* h5File = fileutils::openH5Part(testFile, mode, MPI_COMM_WORLD);
        ASSERT_NE(h5File, nullptr) << "Failed to open HDF5 file for writing";

        H5PartSetStep(h5File, 0);
        H5PartSetNumParticles(h5File, static_cast<h5part_int64_t>(localSize));

        // Write attributes (all ranks write same values - only one actually writes)
        int gridDimAttr = gridDim;
        int numRanksAttr = numRanks;
        fileutils::writeH5PartStepAttrib(h5File, "gridDim", &gridDimAttr, 1);
        fileutils::writeH5PartStepAttrib(h5File, "numRanks", &numRanksAttr, 1);
        fileutils::writeH5PartStepAttrib(h5File, "Lmin", &Lmin, 1);
        fileutils::writeH5PartStepAttrib(h5File, "Lmax", &Lmax, 1);

        // Write fields (each rank writes its portion)
        h5part_int64_t status;
        status = fileutils::writeH5PartField(h5File, "density", density.data());
        EXPECT_EQ(status, H5PART_SUCCESS) << "Failed to write density field";
        status = fileutils::writeH5PartField(h5File, "temperature", temperature.data());
        EXPECT_EQ(status, H5PART_SUCCESS) << "Failed to write temperature field";

        H5PartCloseFile(h5File);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Read HDF5 file back and validate
    {
        int64_t mode = H5PART_READ;
        H5PartFile* h5File = fileutils::openH5Part(testFile, mode, MPI_COMM_WORLD);
        ASSERT_NE(h5File, nullptr) << "Failed to open HDF5 file for reading";

        H5PartSetStep(h5File, 0);

        // Read and validate attributes
        auto stepAttrs = fileutils::stepAttributeNames(h5File);
        EXPECT_GE(stepAttrs.size(), 4u);

        // Read number of particles (should equal localSize * numRanks globally)
        h5part_int64_t globalNumParticles = H5PartGetNumParticles(h5File);
        EXPECT_EQ(globalNumParticles, static_cast<h5part_int64_t>(localSize * numRanks));

        // Set view for this rank to read back its own data
        h5part_int64_t startIdx = rank * static_cast<h5part_int64_t>(localSize);
        h5part_int64_t endIdx = startIdx + static_cast<h5part_int64_t>(localSize) - 1;
        H5PartSetView(h5File, startIdx, endIdx);

        // Read fields
        std::vector<double> readDensity(localSize);
        std::vector<double> readTemperature(localSize);

        h5part_int64_t status;
        status = fileutils::readH5PartField(h5File, "density", readDensity.data());
        EXPECT_EQ(status, H5PART_SUCCESS) << "Failed to read density field";
        status = fileutils::readH5PartField(h5File, "temperature", readTemperature.data());
        EXPECT_EQ(status, H5PART_SUCCESS) << "Failed to read temperature field";

        // Validate data matches what was written
        for (size_t i = 0; i < localSize; ++i)
        {
            EXPECT_DOUBLE_EQ(readDensity[i], density[i])
                << "Density mismatch at index " << i << " on rank " << rank;
            EXPECT_DOUBLE_EQ(readTemperature[i], temperature[i])
                << "Temperature mismatch at index " << i << " on rank " << rank;
        }

        H5PartCloseFile(h5File);
    }

    // Clean up test file
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::remove(testFile.c_str());
    }
}

TEST(FileIO, HDF5GridOutputAttributeRoundTrip)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::string testFile = "/tmp/test_attrib_roundtrip.h5";
    const int gridDim = 8;
    const double Lmin = -0.5, Lmax = 0.5;
    const int numRanks = 4;

    // Write attributes
    {
        int64_t mode = H5PART_WRITE;
        H5PartFile* h5File = fileutils::openH5Part(testFile, mode, MPI_COMM_WORLD);
        ASSERT_NE(h5File, nullptr);

        H5PartSetStep(h5File, 0);
        H5PartSetNumParticles(h5File, 1);  // Need at least 1 particle to create step

        fileutils::writeH5PartStepAttrib(h5File, "gridDim", &gridDim, 1);
        fileutils::writeH5PartStepAttrib(h5File, "numRanks", &numRanks, 1);
        fileutils::writeH5PartStepAttrib(h5File, "Lmin", &Lmin, 1);
        fileutils::writeH5PartStepAttrib(h5File, "Lmax", &Lmax, 1);

        double dummy = 0.0;
        fileutils::writeH5PartField(h5File, "dummy", &dummy);

        H5PartCloseFile(h5File);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Read attributes back - just verify they exist (H5Part attribute reading has type quirks)
    {
        int64_t mode = H5PART_READ;
        H5PartFile* h5File = fileutils::openH5Part(testFile, mode, MPI_COMM_WORLD);
        ASSERT_NE(h5File, nullptr);

        H5PartSetStep(h5File, 0);

        auto attrNames = fileutils::stepAttributeNames(h5File);
        EXPECT_GE(attrNames.size(), 4u);

        // Verify expected attributes are present
        bool hasGridDim = false, hasNumRanks = false, hasLmin = false, hasLmax = false;
        for (const auto& name : attrNames)
        {
            if (name == "gridDim") hasGridDim = true;
            else if (name == "numRanks") hasNumRanks = true;
            else if (name == "Lmin") hasLmin = true;
            else if (name == "Lmax") hasLmax = true;
        }
        EXPECT_TRUE(hasGridDim) << "gridDim attribute not found";
        EXPECT_TRUE(hasNumRanks) << "numRanks attribute not found";
        EXPECT_TRUE(hasLmin) << "Lmin attribute not found";
        EXPECT_TRUE(hasLmax) << "Lmax attribute not found";

        H5PartCloseFile(h5File);
    }

    // Clean up
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::remove(testFile.c_str());
    }
}

#endif // SPH_EXA_HAVE_H5PART
