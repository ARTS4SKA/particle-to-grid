#include <vector>
#include <limits>
#include <numbers>
#include <cmath>
#include <cassert>

#include "cstone/domain/domain.hpp"

using KeyType = uint64_t;

struct DataSender
{
    // vectors to send to each rank in all_to_allv
    std::vector<uint64_t> send_index;
    std::vector<double>   send_dens;
};

template<typename T>
class Mesh
{
public:
    int rank_;
    int numRanks_;
    int gridDim_;
    T   Lmin_;
    T   Lmax_;

    // coordinate centers in the mesh
    std::vector<T> x_;
    // density field per mesh cell
    std::vector<T> dens_;

    // communication counters
    std::vector<int> send_disp;  //(numRanks_+1, 0);
    std::vector<int> send_count; //(numRanks_, 0);

    std::vector<int> recv_disp;  //(numRanks_+1, 0);
    std::vector<int> recv_count; //(numRanks_, 0);

    std::vector<DataSender> vdataSender;

    // vectors to receive from each rank in all_to_allv
    std::vector<uint64_t> send_index;
    std::vector<T>        send_dens;

    // vectors to receive from each rank in all_to_allv
    std::vector<uint64_t> recv_index;
    std::vector<T>        recv_dens;

    // sim box [Lmin, Lmax] - default [0, 1] to match read_pkdgrav.py
    Mesh(int rank, int numRanks, int gridDim, T Lmin = T(0), T Lmax = T(1))
        : rank_(rank)
        , numRanks_(numRanks)
        , gridDim_(gridDim)
        , Lmin_(Lmin)
        , Lmax_(Lmax)
    {
        initCartesianGrid();
    }

    void rasterize_particles_to_mesh(std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y, std::vector<T> z,
                                     std::vector<T> mass)
    {
        std::cout << "rank" << rank_ << " rasterize (density) start" << std::endl;
        std::cout << "rank" << rank_ << " keys between " << *keys.begin() << " - " << *keys.end() << std::endl;

        int particleIndex = 0;
        // iterate over keys vector
        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            auto crd    = calculateKeyIndices(*it, gridDim_);
            int  indexi = std::get<0>(crd);
            int  indexj = std::get<1>(crd);
            int  indexk = std::get<2>(crd);

            assert(indexi < gridDim_);
            assert(indexj < gridDim_);
            assert(indexk < gridDim_);

            // each particle contributes its weight (mass) to its mesh cell
            T contribution = mass[particleIndex];
            assignDensityByMeshCoord(indexi, indexj, indexk, contribution);
            particleIndex++;
        }
        x.clear();
        y.clear();
        z.clear();
        mass.clear();
        keys.clear();

        std::cout << "rank = " << rank_ << " particleIndex = " << particleIndex << std::endl;
        for (int i = 0; i < numRanks_; i++)
            std::cout << "rank = " << rank_ << " send_count = " << send_count[i] << std::endl;

        MPI_Alltoall(send_count.data(), 1, MpiType<int>{}, recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

        for (int i = 0; i < numRanks_; i++)
        {
            send_disp[i + 1] = send_disp[i] + send_count[i];
            recv_disp[i + 1] = recv_disp[i] + recv_count[i];
        }

        // prepare send buffers
        send_index.resize(send_disp[numRanks_]);
        send_dens.resize(send_disp[numRanks_]);
        std::cout << "rank = " << rank_ << " buffers allocated" << std::endl;

        for (int i = 0; i < numRanks_; i++)
        {
            for (int j = send_disp[i]; j < send_disp[i + 1]; j++)
            {
                send_index[j]    = vdataSender[i].send_index[j - send_disp[i]];
                send_dens[j]     = vdataSender[i].send_dens[j - send_disp[i]];
            }
        }
        std::cout << "rank = " << rank_ << " buffers transformed" << std::endl;

        // prepare receive buffers
        recv_index.resize(recv_disp[numRanks_]);
        recv_dens.resize(recv_disp[numRanks_]);

        MPI_Alltoallv(send_index.data(), send_count.data(), send_disp.data(), MpiType<uint64_t>{}, recv_index.data(),
                      recv_count.data(), recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_dens.data(), send_count.data(), send_disp.data(), MpiType<T>{}, recv_dens.data(),
                      recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        std::cout << "rank = " << rank_ << " alltoallv done!" << std::endl;

        for (int i = 0; i < recv_disp[numRanks_]; i++)
        {
            uint64_t index = recv_index[i];
            // accumulate contributions from all ranks into density field
            dens_[index] += recv_dens[i];
        }

        // clear the vectors
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSender[i].send_index.clear();
            vdataSender[i].send_dens.clear();
        }

        convertMassToDensity();
    }

    void assignDensityByMeshCoord(int meshiIndex, int meshjIndex, int meshkIndex, T densContribution)
    {
        int      targetRank  = calculateRankFromMeshCoord(meshkIndex);
        uint64_t targetIndex = calculateInboxIndexFromMeshCoord(meshiIndex, meshjIndex, meshkIndex);

        if (targetRank == rank_)
        {
            // if the corresponding mesh cell belongs to this rank
            uint64_t index = targetIndex;
            if (index >= dens_.size())
            {
                std::cout << "rank = " << rank_ << " index = " << index << " size " << dens_.size() << std::endl;
            }

            // local accumulation into density field
            dens_[index] += densContribution;
        }
        else
        {
            // if the corresponding mesh cell belongs another rank
            send_count[targetRank]++;
            vdataSender[targetRank].send_index.push_back(targetIndex);
            vdataSender[targetRank].send_dens.push_back(densContribution);
        }
    }

    // Convert gridded mass to density by dividing by cell volume
    void convertMassToDensity()
    {
        T cellSize = (Lmax_ - Lmin_) / static_cast<T>(gridDim_);
        T cellVolume = cellSize * cellSize * cellSize;

    #pragma omp parallel for
        for (size_t i = 0; i < dens_.size(); ++i)
        {
            dens_[i] /= cellVolume;
        }
        // std::cout << "rank " << rank_ << " convertMassToDensity done" << std::endl;
    }

    void setSimBox(T Lmin, T Lmax)
    {
        Lmin_ = Lmin;
        Lmax_ = Lmax;
    }

    void resize_comm_size(const int size)
    {
        send_disp.resize(size + 1, 0);
        send_count.resize(size, 0);
        recv_disp.resize(size + 1, 0);
        recv_count.resize(size, 0);

        vdataSender.resize(size);
    }

    inline int calculateRankFromMeshCoord(int k)
    {
        // Rank number is can be calculated as such due to 1D slab decomposition along z-direction
        return k / (gridDim_ / numRanks_);
    }

    inline int calculateInboxIndexFromMeshCoord(int i, int j, int k)
    {
        int base = gridDim_ / numRanks_;
        int remI = i % base;
        int remJ = j % base;
        int remK = k % base;

        int index = remI + remJ * base + remK * base * base;
        // index out of bounds check
        if (index >= gridDim_ * gridDim_ * base)
        {
            std::cerr << "rank " << rank_ << " ERROR: index = " << index << " is out of range" << std::endl;
            return 0;
        }
        return index;
    }

    std::tuple<int, int, int> calculateKeyIndices(KeyType key, int gridDim)
    {
        auto mesh_indices = cstone::decodeHilbert(key);
        // unsigned divisor      = std::pow(2, (21 - powerDim));
        unsigned divisor = 1 + std::pow(2, 21) / gridDim;

        int meshCoordX_base = util::get<0>(mesh_indices) / divisor;
        int meshCoordY_base = util::get<1>(mesh_indices) / divisor;
        int meshCoordZ_base = util::get<2>(mesh_indices) / divisor;

        // std::cout << "key: " << key << " mesh indices: " << meshCoordX_base << " " << meshCoordY_base << " " <<
        // meshCoordZ_base << std::endl;
        return std::tie(meshCoordX_base, meshCoordY_base, meshCoordZ_base);
    }

private:
    size_t  localSize_{0}; // number of mesh cells on this rank

    void initCartesianGrid()
    {
        // compute this rank's z-slab
        localSize_ = static_cast<size_t>(gridDim_) * static_cast<size_t>(gridDim_) * static_cast<size_t>(gridDim_ / numRanks_);
        dens_.assign(localSize_, T{0});
        resize_comm_size(numRanks_);
        setCoordinates(Lmin_, Lmax_);
    }

    // Calculates the volume centers instead of starting with Lmin and adding deltaMesh
    void setCoordinates(T Lmin, T Lmax)
    {
        T deltaMesh     = (Lmax - Lmin) / (gridDim_);
        T centerCoord   = deltaMesh / 2;
        T startingCoord = Lmin + centerCoord;

        x_.resize(gridDim_);

#pragma omp parallel for
        for (int i = 0; i < gridDim_; i++)
        {
            x_[i] = startingCoord + deltaMesh * i;
        }
    }
};

// Forward declaration for CUDA rasterization function
#ifdef USE_CUDA
template<typename T>
void rasterize_particles_to_mesh_cuda(Mesh<T>& mesh, std::vector<KeyType> keys, std::vector<T> x, std::vector<T> y,
                                      std::vector<T> z, std::vector<T> mass);
#endif