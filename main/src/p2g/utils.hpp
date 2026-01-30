#pragma once

#include <chrono>
#include <ostream>
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

class Timer
{
    using Clock = std::chrono::high_resolution_clock;
    using Time  = std::chrono::duration<float>;

public:
    explicit Timer(std::ostream& out) : out(out) {}

    void start()
    {
        tstart = tlast = Clock::now();
        t0 = tstart;
    }

    /** Returns elapsed seconds since last start() or elapsed() call, and prints to stream. */
    float elapsed(const std::string& label)
    {
        tlast = Clock::now();
        float sec = std::chrono::duration_cast<Time>(tlast - tstart).count();
        out << label << " elapsed time: " << sec << " s" << std::endl;
        tstart = tlast;
        return sec;
    }

    /** Returns total seconds since first start(). */
    float totalElapsed() const
    {
        return std::chrono::duration_cast<Time>(Clock::now() - t0).count();
    }

private:
    std::ostream&     out;
    Clock::time_point t0, tstart, tlast;
};

} // namespace p2g
