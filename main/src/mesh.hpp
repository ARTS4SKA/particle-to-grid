#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/mpi_wrappers.hpp"
#include "cstone/sfc/hilbert.hpp"

namespace p2g {

using KeyType = uint64_t;

// Cell (i, j, k) of a gridDim^3 mesh containing the given Hilbert key.
HOST_DEVICE_FUN inline void keyToCell(KeyType key, int gridDim, int& i, int& j, int& k)
{
    auto     xyz     = cstone::decodeHilbert(key);
    unsigned divisor = 1u + (1u << 21) / static_cast<unsigned>(gridDim);
    i                = util::get<0>(xyz) / divisor;
    j                = util::get<1>(xyz) / divisor;
    k                = util::get<2>(xyz) / divisor;
}

// z-slab decomposition: rank r owns planes k in [r*slab, (r+1)*slab), slab = gridDim/numRanks.
// run() guarantees gridDim % numRanks == 0; the clamp only guards the last plane otherwise.
HOST_DEVICE_FUN inline int slabRank(int k, int gridDim, int numRanks)
{
    int r = k / (gridDim / numRanks);
    return r < numRanks ? r : numRanks - 1;
}

// Flat index of global cell (i, j, k) inside the slab of its owning rank.
HOST_DEVICE_FUN inline uint64_t slabLocalIndex(int i, int j, int k, int gridDim, int numRanks)
{
    uint64_t localK = k - slabRank(k, gridDim, numRanks) * (gridDim / numRanks);
    return i + static_cast<uint64_t>(j) * gridDim + localK * gridDim * gridDim;
}

template<typename T>
class CartesianGrid;

#ifdef USE_CUDA
template<typename T>
void cuda_p2g(CartesianGrid<T>& grid, const std::vector<KeyType>& keys,
              const std::vector<std::vector<T>*>& field_ptrs);
#endif

// Uniform Cartesian mesh, decomposed into z-slabs across MPI ranks, with cell-average
// particle<->grid transfer:
//   p2g(keys, fields):   grid_fields_[f][c] = mean of fields[f][p] over particles p in cell c
//   g2p(keys, fields):   fields[f][p] = grid_fields_[f][cell(p)], requires a prior p2g()
// keys are the particles' Hilbert keys (from cstone::Domain::sync).
template<typename T>
class CartesianGrid
{
public:
    int rank_;
    int numRanks_;
    int gridDim_;
    T   Lmin_;
    T   Lmax_;

    std::vector<std::vector<T>> grid_fields_; // [field][local cell]
    std::vector<int>            cell_counts_; // particles per local cell

    // Contributions to cells owned by other ranks, staged per destination rank.
    std::vector<std::vector<uint64_t>>       send_index_;  // [rank][entry]
    std::vector<std::vector<std::vector<T>>> send_values_; // [rank][field][entry]

    CartesianGrid(int rank, int numRanks, int gridDim, T Lmin = T(0), T Lmax = T(1))
        : rank_(rank)
        , numRanks_(numRanks)
        , gridDim_(gridDim)
        , Lmin_(Lmin)
        , Lmax_(Lmax)
        , send_index_(numRanks)
        , send_values_(numRanks)
        , localSize_(static_cast<size_t>(gridDim) * gridDim * (gridDim / numRanks))
    {
        ensureNumFields(1);
    }

    size_t localSize() const { return localSize_; }
    size_t numFields() const { return grid_fields_.size(); }

    std::vector<T>&       dens() { return grid_fields_[0]; }
    const std::vector<T>& dens() const { return grid_fields_[0]; }

    void ensureNumFields(size_t n)
    {
        if (grid_fields_.size() < n) grid_fields_.resize(n, std::vector<T>(localSize_, T(0)));
        for (auto& perField : send_values_)
            if (perField.size() < n) perField.resize(n);
    }

    void p2g(const std::vector<KeyType>& keys, const std::vector<std::vector<T>*>& field_ptrs)
    {
        ensureNumFields(field_ptrs.size());
#ifdef USE_CUDA
        cuda_p2g(*this, keys, field_ptrs);
#else
        const size_t nFields = field_ptrs.size();
        resetAccumulation(0, nFields);

        for (size_t p = 0; p < keys.size(); ++p)
        {
            int i, j, k;
            keyToCell(keys[p], gridDim_, i, j, k);
            int      r   = slabRank(k, gridDim_, numRanks_);
            uint64_t idx = slabLocalIndex(i, j, k, gridDim_, numRanks_);
            assert(idx < localSize_);

            if (r == rank_)
            {
                for (size_t f = 0; f < nFields; ++f)
                    grid_fields_[f][idx] += (*field_ptrs[f])[p];
                cell_counts_[idx]++;
            }
            else
            {
                send_index_[r].push_back(idx);
                for (size_t f = 0; f < nFields; ++f)
                    send_values_[r][f].push_back((*field_ptrs[f])[p]);
            }
        }
        exchangeAndAverage(0, nFields);
#endif
    }

    void g2p(const std::vector<KeyType>& keys, std::vector<std::vector<T>*>& out_field_ptrs)
    {
        const size_t nFields = out_field_ptrs.size();

        std::vector<std::vector<uint64_t>> reqCell(numRanks_);
        std::vector<std::vector<size_t>>   reqParticle(numRanks_);

        for (size_t p = 0; p < keys.size(); ++p)
        {
            int i, j, k;
            keyToCell(keys[p], gridDim_, i, j, k);
            int      r   = slabRank(k, gridDim_, numRanks_);
            uint64_t idx = slabLocalIndex(i, j, k, gridDim_, numRanks_);

            if (r == rank_)
            {
                for (size_t f = 0; f < nFields; ++f)
                    (*out_field_ptrs[f])[p] = grid_fields_[f][idx];
            }
            else
            {
                reqCell[r].push_back(idx);
                reqParticle[r].push_back(p);
            }
        }

        // Round 1: send cell indices to their owners. Round 2: owners reply with the values.
        std::vector<int> reqCounts = bucketSizes(reqCell);
        std::vector<int> srvCounts = alltoallCounts(reqCounts);
        auto             srvCell   = alltoallv(flatten(reqCell), reqCounts, srvCounts, 1);

        std::vector<T> reply(srvCell.size() * nFields);
        for (size_t j = 0; j < srvCell.size(); ++j)
            for (size_t f = 0; f < nFields; ++f)
                reply[j * nFields + f] = grid_fields_[f][srvCell[j]];

        auto   answers = alltoallv(reply, srvCounts, reqCounts, nFields);
        size_t pos     = 0;
        for (int r = 0; r < numRanks_; ++r)
            for (size_t p : reqParticle[r])
                for (size_t f = 0; f < nFields; ++f)
                    (*out_field_ptrs[f])[p] = answers[pos++];
    }

    // The two steps below bracket the per-particle binning; the CUDA backend calls them
    // once per field, the CPU path once for all fields.

    void resetAccumulation(size_t firstField, size_t nFields)
    {
        for (size_t f = firstField; f < firstField + nFields; ++f)
            std::fill(grid_fields_[f].begin(), grid_fields_[f].end(), T(0));
        cell_counts_.assign(localSize_, 0);
        for (int r = 0; r < numRanks_; ++r)
        {
            send_index_[r].clear();
            for (size_t f = firstField; f < firstField + nFields; ++f)
                send_values_[r][f].clear();
        }
    }

    // Deliver staged remote contributions to their owners, then turn sums into cell means.
    void exchangeAndAverage(size_t firstField, size_t nFields)
    {
        std::vector<T> sendValues;
        for (int r = 0; r < numRanks_; ++r)
            for (size_t j = 0; j < send_index_[r].size(); ++j)
                for (size_t f = firstField; f < firstField + nFields; ++f)
                    sendValues.push_back(send_values_[r][f][j]);

        std::vector<int> sendCounts = bucketSizes(send_index_);
        std::vector<int> recvCounts = alltoallCounts(sendCounts);
        auto             recvIndex  = alltoallv(flatten(send_index_), sendCounts, recvCounts, 1);
        auto             recvValues = alltoallv(sendValues, sendCounts, recvCounts, nFields);

        for (size_t j = 0; j < recvIndex.size(); ++j)
        {
            for (size_t f = 0; f < nFields; ++f)
                grid_fields_[firstField + f][recvIndex[j]] += recvValues[j * nFields + f];
            cell_counts_[recvIndex[j]]++;
        }

#pragma omp parallel for
        for (size_t c = 0; c < localSize_; ++c)
            if (cell_counts_[c] > 0)
                for (size_t f = firstField; f < firstField + nFields; ++f)
                    grid_fields_[f][c] /= static_cast<T>(cell_counts_[c]);
    }

private:
    size_t localSize_;

    template<class V>
    static std::vector<int> bucketSizes(const std::vector<std::vector<V>>& buckets)
    {
        std::vector<int> sizes;
        for (const auto& b : buckets)
            sizes.push_back(static_cast<int>(b.size()));
        return sizes;
    }

    template<class V>
    static std::vector<V> flatten(const std::vector<std::vector<V>>& buckets)
    {
        std::vector<V> flat;
        for (const auto& b : buckets)
            flat.insert(flat.end(), b.begin(), b.end());
        return flat;
    }

    std::vector<int> alltoallCounts(const std::vector<int>& sendCounts) const
    {
        std::vector<int> recvCounts(numRanks_);
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        return recvCounts;
    }

    // Alltoallv where rank r sends sendCounts[r] entries of `stride` values each.
    template<class V>
    std::vector<V> alltoallv(const std::vector<V>& send, const std::vector<int>& sendCounts,
                             const std::vector<int>& recvCounts, size_t stride) const
    {
        std::vector<int> sc(numRanks_), rc(numRanks_), sd(numRanks_ + 1, 0), rd(numRanks_ + 1, 0);
        for (int r = 0; r < numRanks_; ++r)
        {
            sc[r]     = sendCounts[r] * static_cast<int>(stride);
            rc[r]     = recvCounts[r] * static_cast<int>(stride);
            sd[r + 1] = sd[r] + sc[r];
            rd[r + 1] = rd[r] + rc[r];
        }
        std::vector<V> recv(rd[numRanks_]);
        MPI_Alltoallv(send.data(), sc.data(), sd.data(), MpiType<V>{}, recv.data(), rc.data(), rd.data(),
                      MpiType<V>{}, MPI_COMM_WORLD);
        return recv;
    }
};

} // namespace p2g
