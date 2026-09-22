#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <string>
#include <tuple>

#include <omp.h>
#include <mpi.h>

namespace p2g {

inline std::tuple<int, int> initMpi()
{
    int rank     = 0;
    int numRanks = 0;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (rank == 0)
    {
        int mpi_version, mpi_subversion;
        MPI_Get_version(&mpi_version, &mpi_subversion);
#ifdef _OPENMP
        printf("# %d MPI-%d.%d process(es) with %d OpenMP-%u thread(s)/process\n", numRanks, mpi_version,
               mpi_subversion, omp_get_max_threads(), _OPENMP);
#else
        printf("# %d MPI-%d.%d process(es) without OpenMP\n", numRanks, mpi_version, mpi_subversion);
#endif
    }
    return std::make_tuple(rank, numRanks);
}

inline int exitSuccess()
{
    MPI_Finalize();
    return EXIT_SUCCESS;
}

inline int exitFailure()
{
    MPI_Finalize();
    return EXIT_FAILURE;
}

class Timer
{
    using Clock = std::chrono::high_resolution_clock;
    using Time  = std::chrono::duration<float>;

public:
    explicit Timer(std::ostream& out)
        : out(out)
    {
    }

    void start() { t0 = tlap = Clock::now(); }

    // Seconds since the previous start()/elapsed() call; also printed with the label.
    float elapsed(const std::string& label)
    {
        auto  now = Clock::now();
        float sec = std::chrono::duration_cast<Time>(now - tlap).count();
        out << label << " elapsed time: " << sec << " s" << std::endl;
        tlap = now;
        return sec;
    }

    float totalElapsed() const { return std::chrono::duration_cast<Time>(Clock::now() - t0).count(); }

private:
    std::ostream&     out;
    Clock::time_point t0, tlap;
};

} // namespace p2g
