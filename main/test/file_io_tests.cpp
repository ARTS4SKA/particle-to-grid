#include <mpi.h>
#include <gtest/gtest.h>
#include <filesystem>
#include <string>

#include "ifile_io_impl.h"

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
