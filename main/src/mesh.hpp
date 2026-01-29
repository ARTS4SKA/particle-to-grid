#pragma once

#include <vector>
#include <limits>
#include <numbers>
#include <cmath>
#include <cassert>
#include <iostream>

#include "cstone/domain/domain.hpp"
#include "p2g/types.hpp"

namespace p2g {

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

    std::vector<DataSender<T>> vdataSender;

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
        std::cout << "rank " << rank_ << " rasterize (nearest_neighbor) start" << std::endl;
        std::cout << "rank " << rank_ << " keys between " << *keys.begin() << " - " << *keys.end() << std::endl;
        resetCommAndDens();
        int particleIndex = 0;
        for (auto it = keys.begin(); it != keys.end(); ++it)
        {
            auto crd    = calculateKeyIndices(*it, gridDim_);
            int  indexi = std::get<0>(crd);
            int  indexj = std::get<1>(crd);
            int  indexk = std::get<2>(crd);
            assert(indexi < gridDim_);
            assert(indexj < gridDim_);
            assert(indexk < gridDim_);
            assignDensityByMeshCoord(indexi, indexj, indexk, mass[particleIndex]);
            particleIndex++;
        }
        performExchangeAndAccumulate();
        convertMassToDensity();
        std::cout << "rank " << rank_ << " rasterize (nearest_neighbor) done" << std::endl;
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

    void resetCommAndDens()
    {
        if (dens_.size() != localSize_) dens_.assign(localSize_, T{0});
        else std::fill(dens_.begin(), dens_.end(), T(0));
        std::fill(send_count.begin(), send_count.end(), 0);
        std::fill(send_disp.begin(), send_disp.end(), 0);
        std::fill(recv_count.begin(), recv_count.end(), 0);
        std::fill(recv_disp.begin(), recv_disp.end(), 0);
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSender[i].send_index.clear();
            vdataSender[i].send_dens.clear();
        }
    }

    void performExchangeAndAccumulate()
    {
        MPI_Alltoall(send_count.data(), 1, MpiType<int>{}, recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);
        for (int i = 0; i < numRanks_; i++)
        {
            send_disp[i + 1] = send_disp[i] + send_count[i];
            recv_disp[i + 1] = recv_disp[i] + recv_count[i];
        }
        send_index.resize(send_disp[numRanks_]);
        send_dens.resize(send_disp[numRanks_]);
        for (int i = 0; i < numRanks_; i++)
            for (int j = send_disp[i]; j < send_disp[i + 1]; j++)
            {
                send_index[j] = vdataSender[i].send_index[j - send_disp[i]];
                send_dens[j]  = vdataSender[i].send_dens[j - send_disp[i]];
            }
        recv_index.resize(recv_disp[numRanks_]);
        recv_dens.resize(recv_disp[numRanks_]);
        MPI_Alltoallv(send_index.data(), send_count.data(), send_disp.data(), MpiType<uint64_t>{},
                      recv_index.data(), recv_count.data(), recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(send_dens.data(), send_count.data(), send_disp.data(), MpiType<T>{},
                      recv_dens.data(), recv_count.data(), recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        for (int i = 0; i < recv_disp[numRanks_]; i++)
            dens_[recv_index[i]] += recv_dens[i];
        for (int i = 0; i < numRanks_; i++)
        {
            vdataSender[i].send_index.clear();
            vdataSender[i].send_dens.clear();
        }
    }

    // Map physical position to cell indices (cell that contains the point). Clamp to [0, gridDim_-1].
    std::tuple<int, int, int> positionToCell(T px, T py, T pz) const
    {
        T dx = (Lmax_ - Lmin_) / static_cast<T>(gridDim_);
        int i = static_cast<int>((px - Lmin_) / dx);
        int j = static_cast<int>((py - Lmin_) / dx);
        int k = static_cast<int>((pz - Lmin_) / dx);
        i = std::max(0, std::min(i, gridDim_ - 1));
        j = std::max(0, std::min(j, gridDim_ - 1));
        k = std::max(0, std::min(k, gridDim_ - 1));
        return std::tie(i, j, k);
    }

    // Get physical coordinates of cell center (i, j, k).
    void cellCenter(int i, int j, int k, T& cx, T& cy, T& cz) const
    {
        T dx   = (Lmax_ - Lmin_) / static_cast<T>(gridDim_);
        T half = dx * T(0.5);
        cx = Lmin_ + half + dx * static_cast<T>(i);
        cy = Lmin_ + half + dx * static_cast<T>(j);
        cz = Lmin_ + half + dx * static_cast<T>(k);
    }

    // Cell-averaged rasterization: assign each particle to the cell containing it by position.
    void rasterize_particles_to_mesh_cell_average(std::vector<T> x, std::vector<T> y, std::vector<T> z,
                                                   std::vector<T> mass)
    {
        std::cout << "rank " << rank_ << " rasterize (cell_average) start" << std::endl;
        resetCommAndDens();
        for (size_t p = 0; p < x.size(); ++p)
        {
            auto [ii, jj, kk] = positionToCell(x[p], y[p], z[p]);
            assignDensityByMeshCoord(ii, jj, kk, mass[p]);
        }
        performExchangeAndAccumulate();
        convertMassToDensity();
        std::cout << "rank " << rank_ << " rasterize (cell_average) done" << std::endl;
    }

    // SPH kernel: 3D cubic spline, support 2h. W(r,h) = (sigma/h^3) * f(q), q = r/h.
    static T sphKernel(T r, T h)
    {
        if (h <= T(0) || r > T(2) * h) return T(0);
        T q = r / h;
        T sigma = T(8) / (static_cast<T>(std::numbers::pi));
        T fac = sigma / (h * h * h);
        if (q <= T(1))
            return fac * (T(1) - T(1.5) * q * q + T(0.75) * q * q * q);
        else
            return fac * T(0.25) * (T(2) - q) * (T(2) - q) * (T(2) - q);
    }

    // SPH rasterization: each particle contributes to all cells within 2h with kernel weight.
    void rasterize_particles_to_mesh_sph(std::vector<T> x, std::vector<T> y, std::vector<T> z,
                                         const std::vector<T>& h, std::vector<T> mass)
    {
        std::cout << "rank " << rank_ << " rasterize (sph) start" << std::endl;
        resetCommAndDens();
        T dx = (Lmax_ - Lmin_) / static_cast<T>(gridDim_);
        for (size_t p = 0; p < x.size(); ++p)
        {
            T hp = h[p];
            if (hp <= T(0)) continue;
            T xp = x[p], yp = y[p], zp = z[p];
            int supportCells = static_cast<int>(std::ceil(T(2) * hp / dx)) + 1;
            auto [i0, j0, k0] = positionToCell(xp, yp, zp);
            for (int di = -supportCells; di <= supportCells; ++di)
            {
                for (int dj = -supportCells; dj <= supportCells; ++dj)
                {
                    for (int dk = -supportCells; dk <= supportCells; ++dk)
                    {
                        int i = i0 + di, j = j0 + dj, k = k0 + dk;
                        if (i < 0 || i >= gridDim_ || j < 0 || j >= gridDim_ || k < 0 || k >= gridDim_) continue;
                        T cx, cy, cz;
                        cellCenter(i, j, k, cx, cy, cz);
                        T r = std::sqrt((xp - cx) * (xp - cx) + (yp - cy) * (yp - cy) + (zp - cz) * (zp - cz));
                        T w = sphKernel(r, hp);
                        if (w > T(0)) assignDensityByMeshCoord(i, j, k, mass[p] * w);
                    }
                }
            }
        }
        performExchangeAndAccumulate();
        // SPH deposits density directly (mass*W has units 1/length^3); do not divide by cell volume
        std::cout << "rank " << rank_ << " rasterize (sph) done" << std::endl;
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

} // namespace p2g

// Forward declarations for CUDA rasterization (nearest neighbor, cell average, SPH)
#ifdef USE_CUDA
template<typename T>
void rasterize_particles_to_mesh_cuda(p2g::Mesh<T>& mesh, std::vector<p2g::KeyType> keys, std::vector<T> x,
                                      std::vector<T> y, std::vector<T> z, std::vector<T> mass);
template<typename T>
void rasterize_particles_to_mesh_cuda_cell_average(p2g::Mesh<T>& mesh, std::vector<T> x, std::vector<T> y,
                                                   std::vector<T> z, std::vector<T> mass);
template<typename T>
void rasterize_particles_to_mesh_cuda_sph(p2g::Mesh<T>& mesh, std::vector<T> x, std::vector<T> y, std::vector<T> z,
                                          const std::vector<T>& h, std::vector<T> mass);
#endif