#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
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

// How a field's per-cell contributions are combined.
//   Mean    — arithmetic mean over the particles in the cell. Right for intensive quantities
//             (temperature, velocity): the average of the values present.
//   Density — total of the particle values divided by the cell volume. Right for extensive
//             quantities (mass): summing mass and dividing by volume gives a mass density, and
//             total mass is conserved. Averaging mass instead yields the mean particle mass,
//             which for equal-mass particles is the same constant in every occupied cell.
enum class Reduction { Mean, Density };

// Monaghan M4 cubic spline, the standard SPH kernel, with support radius 2h (the convention
// SPH-EXA uses). W(r,h) = (1/(pi h^3)) f(r/h), normalised so that the integral over space is 1.
//
// Beware the competing convention in which h *is* the support radius and the prefactor is
// 8/(pi h^3) with q = 2r/h. Mixing the two overestimates the kernel eightfold.
template<typename T>
HOST_DEVICE_FUN inline T sphKernel(T r, T h)
{
    if (h <= T(0) || r >= T(2) * h) return T(0);

    const T q   = r / h;
    const T fac = T(0.3183098861837907) / (h * h * h); // 1/pi
    if (q <= T(1)) return fac * (T(1) - T(1.5) * q * q + T(0.75) * q * q * q);
    const T t = T(2) - q;
    return fac * T(0.25) * t * t * t;
}

// Geometry shared by the CPU and CUDA paths. Keeping one copy of each formula is deliberate:
// the key-decoding arithmetic was previously duplicated between mesh.hpp and mesh.cu, which is
// exactly the kind of thing that drifts.

template<typename T>
HOST_DEVICE_FUN inline T cellSizeOf(T lmin, T lmax, int gridDim)
{
    return (lmax - lmin) / static_cast<T>(gridDim);
}

template<typename T>
HOST_DEVICE_FUN inline T cellCenterAt(int i, T lmin, T dx)
{
    return lmin + (static_cast<T>(i) + T(0.5)) * dx;
}

template<typename T>
HOST_DEVICE_FUN inline int cellContaining(T p, T lmin, T dx, int gridDim)
{
    int i = static_cast<int>((p - lmin) / dx);
    return i < 0 ? 0 : (i >= gridDim ? gridDim - 1 : i);
}

// Cell index wrapped into the grid, or -1 when it falls outside a non-periodic box.
HOST_DEVICE_FUN inline int wrapIndex(int i, int gridDim, bool periodic)
{
    if (!periodic) return (i < 0 || i >= gridDim) ? -1 : i;
    return (i % gridDim + gridDim) % gridDim;
}

// Signed distance taking the shorter way round a periodic box.
template<typename T>
HOST_DEVICE_FUN inline T minImage(T d, T boxLength, bool periodic)
{
    if (periodic)
    {
        if (d > T(0.5) * boxLength) d -= boxLength;
        else if (d < T(-0.5) * boxLength) d += boxLength;
    }
    return d;
}

template<typename T>
class CartesianGrid;

#ifdef USE_CUDA
template<typename T>
void cuda_p2g(CartesianGrid<T>& grid, const std::vector<KeyType>& keys,
              const std::vector<std::vector<T>*>& field_ptrs);

template<typename T>
void cuda_p2g_sph(CartesianGrid<T>& grid, const std::vector<T>& x, const std::vector<T>& y,
                  const std::vector<T>& z, const std::vector<T>& h,
                  const std::vector<std::vector<T>*>& field_ptrs, int maxSupportCells, bool periodic);
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

    std::vector<std::vector<T>> grid_fields_;  // [field][local cell]
    std::vector<int>            cell_counts_;  // particles per local cell (nearest-cell Mean)
    std::vector<T>              cell_weights_; // summed kernel weight per local cell (SPH Mean)

    // Contributions to cells owned by other ranks, staged per destination rank.
    std::vector<std::vector<uint64_t>>       send_index_;   // [rank][entry]
    std::vector<std::vector<std::vector<T>>> send_values_;  // [rank][field][entry]
    std::vector<std::vector<T>>              send_weights_; // [rank][entry], SPH only

    CartesianGrid(int rank, int numRanks, int gridDim, T Lmin = T(0), T Lmax = T(1))
        : rank_(rank)
        , numRanks_(numRanks)
        , gridDim_(gridDim)
        , Lmin_(Lmin)
        , Lmax_(Lmax)
        , send_index_(numRanks)
        , send_values_(numRanks)
        , send_weights_(numRanks)
        , localSize_(static_cast<size_t>(gridDim) * gridDim * (gridDim / numRanks))
    {
        ensureNumFields(1);
    }

    size_t localSize() const { return localSize_; }
    size_t numFields() const { return grid_fields_.size(); }

    std::vector<T>&       dens() { return grid_fields_[0]; }
    const std::vector<T>& dens() const { return grid_fields_[0]; }

    T cellSize() const { return cellSizeOf(Lmin_, Lmax_, gridDim_); }

    T cellVolume() const
    {
        const T dx = cellSize();
        return dx * dx * dx;
    }

    // Center of global cell index i along one axis.
    T cellCenter(int i) const { return cellCenterAt(i, Lmin_, cellSize()); }

    // Cell containing position p along one axis, clamped to the grid.
    int positionToCell(T p) const { return cellContaining(p, Lmin_, cellSize(), gridDim_); }

    // Fields default to Mean; set Density on the mass field to get a mass density.
    void      setReduction(size_t field, Reduction r) { reductions_.at(field) = r; }
    Reduction reduction(size_t field) const { return reductions_.at(field); }

    void ensureNumFields(size_t n)
    {
        if (grid_fields_.size() < n)
        {
            // Each field is allocated in place. Passing a filled vector to resize() would build
            // one temporary and copy it into every new slot, doubling peak memory -- 16 GiB
            // instead of 8 for a 1024^3 grid of doubles.
            const size_t first = grid_fields_.size();
            grid_fields_.resize(n);
            for (size_t f = first; f < n; ++f)
                grid_fields_[f].assign(localSize_, T(0));
        }
        if (reductions_.size() < n) reductions_.resize(n, Reduction::Mean);
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
                if (needCounts_) cell_counts_[idx]++;
            }
            else
            {
                send_index_[r].push_back(idx);
                for (size_t f = 0; f < nFields; ++f)
                    send_values_[r][f].push_back((*field_ptrs[f])[p]);
            }
        }
        exchangeAndReduce(0, nFields);
#endif
    }

    // Invertible P2G: the grid still holds cell means, and `residual_ptrs[f][p]` receives
    // v[f][p] - mean(cell(p)), the part the mean alone cannot represent. Passing those residuals
    // back to g2p() reconstructs the input. A cell holding n particles has one mean and n-1
    // independent residuals, so (grid, residuals) carries exactly as many numbers as the input:
    // this is a change of basis, not extra information.
    // Each residual array must have one entry per particle.
    void p2g(const std::vector<KeyType>& keys, const std::vector<std::vector<T>*>& field_ptrs,
             std::vector<std::vector<T>*>& residual_ptrs)
    {
        assert(residual_ptrs.size() == field_ptrs.size());
        ensureNumFields(field_ptrs.size());
        // A residual is v - mean(cell), so the cell has to hold that mean. For a Density field
        // the cell holds mass/volume instead, and the difference would be meaningless.
        for (size_t f = 0; f < field_ptrs.size(); ++f)
            assert(reductions_[f] == Reduction::Mean);
        p2g(keys, field_ptrs);

        // A particle's cell may live on another rank, so the means come back through the g2p path.
        g2p(keys, residual_ptrs);
        for (size_t f = 0; f < residual_ptrs.size(); ++f)
        {
            assert(residual_ptrs[f]->size() == keys.size());
            for (size_t p = 0; p < keys.size(); ++p)
                (*residual_ptrs[f])[p] = (*field_ptrs[f])[p] - (*residual_ptrs[f])[p];
        }
    }

    // Inverse of p2g(keys, fields, residuals): returns mean(cell(p)) + residual[p] == the original
    // value. Without the residuals, g2p() returns only the cell mean.
    void g2p(const std::vector<KeyType>& keys, std::vector<std::vector<T>*>& out_field_ptrs,
             const std::vector<std::vector<T>*>& residual_ptrs)
    {
        assert(residual_ptrs.size() == out_field_ptrs.size());
        g2p(keys, out_field_ptrs);

        for (size_t f = 0; f < out_field_ptrs.size(); ++f)
        {
            assert(residual_ptrs[f]->size() == keys.size());
            for (size_t p = 0; p < keys.size(); ++p)
                (*out_field_ptrs[f])[p] += (*residual_ptrs[f])[p];
        }
    }

    // SPH deposition: each particle spreads its payload over the cells within its kernel support
    // (2h) instead of landing in a single cell, which gives a smooth field whose resolution follows
    // the particles' own smoothing lengths. Needs positions and h, not just Hilbert keys.
    //
    //   Density fields  grid[c] = sum_p v_p w_pc / V_cell
    //   Mean fields     grid[c] = sum_p v_p w_pc / sum_p w_pc   (kernel-weighted average)
    //
    // The per-particle weights are normalised so that sum_c w_pc == 1. That makes the transfer
    // conserve mass exactly, independently of the kernel's own normalisation and of any truncation
    // of its support, which the raw kernel sampled at cell centers does not.
    //
    // Two cases that would otherwise lose mass are handled explicitly:
    //   - h far below the cell size, so no cell center falls inside the support: the particle's
    //     payload goes entirely into the cell containing it, i.e. the nearest-cell behaviour.
    //     Cosmological runs hit this constantly, where h in a halo is far smaller than a cell.
    //   - support wider than max_support_cells cells: the support is truncated and the weights
    //     renormalised, which is equivalent to shrinking h for that particle.
    //
    // Runs on the GPU when built with USE_CUDA, otherwise on the CPU. p2gSphCpu() forces the
    // host path, which is what the CPU/GPU agreement test compares against.
    void p2gSph(const std::vector<T>& x, const std::vector<T>& y, const std::vector<T>& z,
                const std::vector<T>& h, const std::vector<std::vector<T>*>& field_ptrs,
                int maxSupportCells = 8, bool periodic = true)
    {
#ifdef USE_CUDA
        ensureNumFields(field_ptrs.size());
        cuda_p2g_sph(*this, x, y, z, h, field_ptrs, maxSupportCells, periodic);
#else
        p2gSphCpu(x, y, z, h, field_ptrs, maxSupportCells, periodic);
#endif
    }

    void p2gSphCpu(const std::vector<T>& x, const std::vector<T>& y, const std::vector<T>& z,
                   const std::vector<T>& h, const std::vector<std::vector<T>*>& field_ptrs,
                   int maxSupportCells = 8, bool periodic = true)
    {
        const size_t nFields = field_ptrs.size();
        const size_t nPart   = x.size();
        assert(y.size() == nPart && z.size() == nPart && h.size() == nPart);

        ensureNumFields(nFields);
        resetSphAccumulation(0, nFields);

        const T dx = cellSize();
        // Keeping the stencil narrower than the grid stops periodic wrapping from folding one
        // cell into the stencil twice, which would double-count its weight.
        const int maxStencil = std::min(maxSupportCells, (gridDim_ - 1) / 2);

        std::vector<int> ci, cj, ck;
        std::vector<T>   weight;

        for (size_t p = 0; p < nPart; ++p)
        {
            const T xp = x[p], yp = y[p], zp = z[p], hp = h[p];

            const int i0 = positionToCell(xp);
            const int j0 = positionToCell(yp);
            const int k0 = positionToCell(zp);

            ci.clear();
            cj.clear();
            ck.clear();
            weight.clear();

            T sumW = T(0);
            if (hp > T(0))
            {
                const int stencil = std::min(maxStencil, static_cast<int>(std::ceil(T(2) * hp / dx)));
                for (int di = -stencil; di <= stencil; ++di)
                    for (int dj = -stencil; dj <= stencil; ++dj)
                        for (int dk = -stencil; dk <= stencil; ++dk)
                        {
                            const int i = wrapIndex(i0 + di, gridDim_, periodic);
                            const int j = wrapIndex(j0 + dj, gridDim_, periodic);
                            const int k = wrapIndex(k0 + dk, gridDim_, periodic);
                            if (i < 0 || j < 0 || k < 0) continue; // outside a non-periodic box

                            const T boxLength = Lmax_ - Lmin_;
                            const T sx = minImage(xp - cellCenter(i), boxLength, periodic);
                            const T sy = minImage(yp - cellCenter(j), boxLength, periodic);
                            const T sz = minImage(zp - cellCenter(k), boxLength, periodic);
                            const T w  = sphKernel(std::sqrt(sx * sx + sy * sy + sz * sz), hp);
                            if (w <= T(0)) continue;

                            ci.push_back(i);
                            cj.push_back(j);
                            ck.push_back(k);
                            weight.push_back(w);
                            sumW += w;
                        }
            }

            if (sumW <= T(0))
            {
                // Sub-cell smoothing length: fall back to the cell holding the particle.
                ci.assign(1, i0);
                cj.assign(1, j0);
                ck.assign(1, k0);
                weight.assign(1, T(1));
                sumW = T(1);
            }

            const T invSum = T(1) / sumW;
            for (size_t c = 0; c < weight.size(); ++c)
                depositSph(ci[c], cj[c], ck[c], weight[c] * invSum, p, field_ptrs, nFields);
        }

        exchangeAndReduceSph(0, nFields);
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

    // Whether this transfer needs a particle count per cell. Only Mean does; skipping it saves
    // 4 bytes per cell, which is 4 GiB on a 1024^3 grid.
    bool needsCellCounts() const { return needCounts_; }

    // Whether the SPH transfer needs a summed kernel weight per cell. Only Mean does.
    bool needsCellWeights() const { return needWeights_; }

    void resetAccumulation(size_t firstField, size_t nFields)
    {
        for (size_t f = firstField; f < firstField + nFields; ++f)
            std::fill(grid_fields_[f].begin(), grid_fields_[f].end(), T(0));

        needCounts_ = false;
        for (size_t f = firstField; f < firstField + nFields; ++f)
            needCounts_ = needCounts_ || reductions_[f] == Reduction::Mean;

        if (needCounts_) { cell_counts_.assign(localSize_, 0); }
        else
        {
            cell_counts_.clear();
            cell_counts_.shrink_to_fit();
        }
        for (int r = 0; r < numRanks_; ++r)
        {
            send_index_[r].clear();
            for (size_t f = firstField; f < firstField + nFields; ++f)
                send_values_[r][f].clear();
        }
    }

    // Deliver staged remote contributions to their owners, then reduce each field's accumulated
    // sum according to its Reduction.
    void exchangeAndReduce(size_t firstField, size_t nFields)
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
            if (needCounts_) cell_counts_[recvIndex[j]]++;
        }

        const T invVolume = T(1) / cellVolume();
        for (size_t f = firstField; f < firstField + nFields; ++f)
        {
            if (reductions_[f] == Reduction::Density)
            {
                // Mass stays summed; only the volume divides it. Empty cells keep a density of 0.
#pragma omp parallel for
                for (size_t c = 0; c < localSize_; ++c)
                    grid_fields_[f][c] *= invVolume;
            }
            else
            {
#pragma omp parallel for
                for (size_t c = 0; c < localSize_; ++c)
                    if (cell_counts_[c] > 0) grid_fields_[f][c] /= static_cast<T>(cell_counts_[c]);
            }
        }
    }

    // The two SPH stages, exposed so the CUDA backend can drive them one field at a time.
    void resetSphAccumulation(size_t firstField, size_t nFields)
    {
        for (size_t f = firstField; f < firstField + nFields; ++f)
            std::fill(grid_fields_[f].begin(), grid_fields_[f].end(), T(0));

        needCounts_  = false;
        needWeights_ = false;
        for (size_t f = firstField; f < firstField + nFields; ++f)
            needWeights_ = needWeights_ || reductions_[f] == Reduction::Mean;

        cell_counts_.clear();
        cell_counts_.shrink_to_fit();
        if (needWeights_) { cell_weights_.assign(localSize_, T(0)); }
        else
        {
            cell_weights_.clear();
            cell_weights_.shrink_to_fit();
        }

        for (int r = 0; r < numRanks_; ++r)
        {
            send_index_[r].clear();
            send_weights_[r].clear();
            for (size_t f = firstField; f < firstField + nFields; ++f)
                send_values_[r][f].clear();
        }
    }

    void exchangeAndReduceSph(size_t firstField, size_t nFields)
    {
        const size_t   stride = nFields + 1;
        std::vector<T> sendValues;
        for (int r = 0; r < numRanks_; ++r)
            for (size_t j = 0; j < send_index_[r].size(); ++j)
            {
                for (size_t f = firstField; f < firstField + nFields; ++f)
                    sendValues.push_back(send_values_[r][f][j]);
                sendValues.push_back(send_weights_[r][j]);
            }

        std::vector<int> sendCounts = bucketSizes(send_index_);
        std::vector<int> recvCounts = alltoallCounts(sendCounts);
        auto             recvIndex  = alltoallv(flatten(send_index_), sendCounts, recvCounts, 1);
        auto             recvValues = alltoallv(sendValues, sendCounts, recvCounts, stride);

        for (size_t j = 0; j < recvIndex.size(); ++j)
        {
            for (size_t f = 0; f < nFields; ++f)
                grid_fields_[firstField + f][recvIndex[j]] += recvValues[j * stride + f];
            if (needWeights_) cell_weights_[recvIndex[j]] += recvValues[j * stride + nFields];
        }

        const T invVolume = T(1) / cellVolume();
        for (size_t f = firstField; f < firstField + nFields; ++f)
        {
            if (reductions_[f] == Reduction::Density)
            {
#pragma omp parallel for
                for (size_t c = 0; c < localSize_; ++c)
                    grid_fields_[f][c] *= invVolume;
            }
            else
            {
#pragma omp parallel for
                for (size_t c = 0; c < localSize_; ++c)
                    if (cell_weights_[c] > T(0)) grid_fields_[f][c] /= cell_weights_[c];
            }
        }
    }

private:
    size_t                 localSize_;
    std::vector<Reduction> reductions_;
    bool                   needCounts_{true};
    bool                   needWeights_{false};


    // Add one particle's weighted contribution to a cell, locally or via the send buffers.
    void depositSph(int i, int j, int k, T w, size_t p, const std::vector<std::vector<T>*>& field_ptrs,
                    size_t nFields)
    {
        const int      r   = slabRank(k, gridDim_, numRanks_);
        const uint64_t idx = slabLocalIndex(i, j, k, gridDim_, numRanks_);

        if (r == rank_)
        {
            assert(idx < localSize_);
            for (size_t f = 0; f < nFields; ++f)
                grid_fields_[f][idx] += (*field_ptrs[f])[p] * w;
            if (needWeights_) cell_weights_[idx] += w;
        }
        else
        {
            send_index_[r].push_back(idx);
            send_weights_[r].push_back(w);
            for (size_t f = 0; f < nFields; ++f)
                send_values_[r][f].push_back((*field_ptrs[f])[p] * w);
        }
    }

    // Like exchangeAndReduce, but each entry carries its kernel weight alongside the values so
    // that Mean fields can be normalised by the total weight that reached the cell.

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
