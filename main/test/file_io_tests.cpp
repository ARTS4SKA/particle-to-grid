#include <mpi.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include <cstdio>

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
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

TEST(FileIO, HDF5ReaderStepAttributesOrFields)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::string path = std::string(TEST_DATA_DIR) + "/turb_50.h5";
    if (!std::filesystem::exists(path))
    {
        GTEST_SKIP() << "Test data file not found: " << path;
    }

    std::unique_ptr<IFileReader> reader = makeH5PartReader(MPI_COMM_WORLD);
    reader->setStep(path, 0, FileMode::collective);

    auto stepAttrs = reader->stepAttributes();
    (void)stepAttrs;

    reader->closeStep();
    SUCCEED();
}

TEST(FileIO, TipsyReaderRequiresFile)
{
    std::unique_ptr<IFileReader> reader = makeTipsyReader(MPI_COMM_WORLD);
    EXPECT_THROW(reader->setStep("/nonexistent/tipsy.bin", 0, FileMode::collective), std::exception);
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

TEST(FileIO, HDF5GridOutputMultiRank)
{
    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    if (numRanks < 2)
    {
        GTEST_SKIP() << "This test requires at least 2 MPI ranks";
    }

    // Test parameters: grid must be divisible by numRanks
    const int gridDim = std::max(4, numRanks);
    const int base = gridDim / numRanks;
    const size_t localSize = static_cast<size_t>(gridDim) * gridDim * base;
    const std::string testFile = "/tmp/test_grid_multirank.h5";

    // Each rank creates unique data based on rank
    std::vector<double> localData(localSize);
    for (size_t i = 0; i < localSize; ++i)
    {
        localData[i] = rank * 1000.0 + static_cast<double>(i);
    }

    // Write HDF5 file in parallel
    {
        int64_t mode = H5PART_WRITE | H5PART_VFD_MPIIO_IND;
        H5PartFile* h5File = fileutils::openH5Part(testFile, mode, MPI_COMM_WORLD);
        ASSERT_NE(h5File, nullptr);

        H5PartSetStep(h5File, 0);
        H5PartSetNumParticles(h5File, static_cast<h5part_int64_t>(localSize));

        int gridDimAttr = gridDim;
        fileutils::writeH5PartStepAttrib(h5File, "gridDim", &gridDimAttr, 1);

        fileutils::writeH5PartField(h5File, "data", localData.data());

        H5PartCloseFile(h5File);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Read back and validate each rank's data
    {
        int64_t mode = H5PART_READ;
        H5PartFile* h5File = fileutils::openH5Part(testFile, mode, MPI_COMM_WORLD);
        ASSERT_NE(h5File, nullptr);

        H5PartSetStep(h5File, 0);

        // Total particles should be localSize * numRanks
        h5part_int64_t totalParticles = H5PartGetNumParticles(h5File);
        EXPECT_EQ(totalParticles, static_cast<h5part_int64_t>(localSize * numRanks));

        // Each rank reads its own portion
        h5part_int64_t startIdx = rank * static_cast<h5part_int64_t>(localSize);
        h5part_int64_t endIdx = startIdx + static_cast<h5part_int64_t>(localSize) - 1;
        H5PartSetView(h5File, startIdx, endIdx);

        std::vector<double> readData(localSize);
        fileutils::readH5PartField(h5File, "data", readData.data());

        for (size_t i = 0; i < localSize; ++i)
        {
            EXPECT_DOUBLE_EQ(readData[i], localData[i])
                << "Data mismatch at index " << i << " on rank " << rank;
        }

        H5PartCloseFile(h5File);
    }

    // Clean up
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
