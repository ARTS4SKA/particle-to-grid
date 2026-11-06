#include "mesh.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>

namespace p2g {

static void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// One thread per particle: local contributions are accumulated atomically, remote ones are
// appended to a staging buffer that the host routes into the grid's send buffers.
template<class T>
__global__ void cellAverageKernel(const KeyType* keys, const T* values, int numParticles, int gridDim,
                                  int numRanks, int rank, T* grid, int* counts, int* remoteRanks,
                                  uint64_t* remoteIndices, T* remoteValues, int* remoteCount)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numParticles) return;

    int i, j, k;
    keyToCell(keys[p], gridDim, i, j, k);
    if (i < 0 || i >= gridDim || j < 0 || j >= gridDim || k < 0 || k >= gridDim) return;

    int      r   = slabRank(k, gridDim, numRanks);
    uint64_t idx = slabLocalIndex(i, j, k, gridDim, numRanks);

    if (r == rank)
    {
        atomicAdd(&grid[idx], values[p]);
        atomicAdd(&counts[idx], 1);
    }
    else
    {
        int pos            = atomicAdd(remoteCount, 1);
        remoteRanks[pos]   = r;
        remoteIndices[pos] = idx;
        remoteValues[pos]  = values[p];
    }
}

// Remote staging order is nondeterministic across kernel launches, so each field is
// binned and exchanged on its own.
template<typename T>
void cuda_p2g(CartesianGrid<T>& grid, const std::vector<KeyType>& keys,
              const std::vector<std::vector<T>*>& field_ptrs)
{
    const size_t n         = keys.size();
    const size_t localSize = grid.localSize();

    KeyType*  d_keys = nullptr;
    T*        d_values = nullptr, *d_grid = nullptr, *d_remoteValues = nullptr;
    int*      d_counts = nullptr, *d_remoteRanks = nullptr, *d_remoteCount = nullptr;
    uint64_t* d_remoteIdx = nullptr;

    if (n > 0)
    {
        checkCudaError(cudaMalloc(&d_keys, n * sizeof(KeyType)), "d_keys");
        checkCudaError(cudaMalloc(&d_values, n * sizeof(T)), "d_values");
        checkCudaError(cudaMalloc(&d_grid, localSize * sizeof(T)), "d_grid");
        checkCudaError(cudaMalloc(&d_counts, localSize * sizeof(int)), "d_counts");
        checkCudaError(cudaMalloc(&d_remoteRanks, n * sizeof(int)), "d_remoteRanks");
        checkCudaError(cudaMalloc(&d_remoteIdx, n * sizeof(uint64_t)), "d_remoteIdx");
        checkCudaError(cudaMalloc(&d_remoteValues, n * sizeof(T)), "d_remoteValues");
        checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "d_remoteCount");
        checkCudaError(cudaMemcpy(d_keys, keys.data(), n * sizeof(KeyType), cudaMemcpyHostToDevice), "copy keys");
    }

    for (size_t f = 0; f < field_ptrs.size(); ++f)
    {
        grid.resetAccumulation(f, 1);

        if (n > 0)
        {
            const int zero = 0;
            checkCudaError(cudaMemcpy(d_values, field_ptrs[f]->data(), n * sizeof(T), cudaMemcpyHostToDevice),
                           "copy values");
            checkCudaError(cudaMemset(d_grid, 0, localSize * sizeof(T)), "zero grid");
            checkCudaError(cudaMemset(d_counts, 0, localSize * sizeof(int)), "zero counts");
            checkCudaError(cudaMemcpy(d_remoteCount, &zero, sizeof(int), cudaMemcpyHostToDevice), "zero remoteCount");

            const int threadsPerBlock = 256;
            const int blocks          = (static_cast<int>(n) + threadsPerBlock - 1) / threadsPerBlock;
            cellAverageKernel<<<blocks, threadsPerBlock>>>(d_keys, d_values, static_cast<int>(n), grid.gridDim_,
                                                           grid.numRanks_, grid.rank_, d_grid, d_counts,
                                                           d_remoteRanks, d_remoteIdx, d_remoteValues,
                                                           d_remoteCount);
            checkCudaError(cudaDeviceSynchronize(), "cellAverageKernel");

            checkCudaError(cudaMemcpy(grid.grid_fields_[f].data(), d_grid, localSize * sizeof(T),
                                      cudaMemcpyDeviceToHost), "grid back");
            // Density needs no per-cell counts, and the host vector is left unallocated for it.
            if (grid.needsCellCounts())
                checkCudaError(cudaMemcpy(grid.cell_counts_.data(), d_counts, localSize * sizeof(int),
                                          cudaMemcpyDeviceToHost), "counts back");

            int numRemote = 0;
            checkCudaError(cudaMemcpy(&numRemote, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost),
                           "remoteCount back");
            if (numRemote > 0)
            {
                std::vector<int>      ranks(numRemote);
                std::vector<uint64_t> indices(numRemote);
                std::vector<T>        values(numRemote);
                checkCudaError(cudaMemcpy(ranks.data(), d_remoteRanks, numRemote * sizeof(int),
                                          cudaMemcpyDeviceToHost), "remoteRanks back");
                checkCudaError(cudaMemcpy(indices.data(), d_remoteIdx, numRemote * sizeof(uint64_t),
                                          cudaMemcpyDeviceToHost), "remoteIdx back");
                checkCudaError(cudaMemcpy(values.data(), d_remoteValues, numRemote * sizeof(T),
                                          cudaMemcpyDeviceToHost), "remoteValues back");
                for (int m = 0; m < numRemote; ++m)
                {
                    grid.send_index_[ranks[m]].push_back(indices[m]);
                    grid.send_values_[ranks[m]][f].push_back(values[m]);
                }
            }
        }

        grid.exchangeAndReduce(f, 1);
    }

    if (n > 0)
    {
        cudaFree(d_keys);
        cudaFree(d_values);
        cudaFree(d_grid);
        cudaFree(d_counts);
        cudaFree(d_remoteRanks);
        cudaFree(d_remoteIdx);
        cudaFree(d_remoteValues);
        cudaFree(d_remoteCount);
    }
}


// -----------------------------------------------------------------------
// SPH deposition
// -----------------------------------------------------------------------
// One thread per particle. Each thread walks its own stencil twice: once to total the kernel
// weights, once to deposit value*w/sumW. Recomputing the kernel is cheaper than finding somewhere
// to store a per-thread cell list, and keeps the thread's memory use constant.
//
// The geometry helpers come from mesh.hpp so that host and device evaluate the same formulas.

template<class T>
__global__ void sphDepositKernel(const T* x, const T* y, const T* z, const T* hArr, const T* values,
                                 int numParticles, int gridDim, int numRanks, int rank, T lmin, T lmax,
                                 int maxSupportCells, bool periodic, T* grid, T* weights, bool needWeights,
                                 int* remoteRanks, uint64_t* remoteIndices, T* remoteValues, T* remoteWeights,
                                 int* remoteCount, int remoteCapacity)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= numParticles) return;

    const T dx        = cellSizeOf(lmin, lmax, gridDim);
    const T boxLength = lmax - lmin;
    const T xp = x[p], yp = y[p], zp = z[p], hp = hArr[p];
    const T value = values[p];

    const int i0 = cellContaining(xp, lmin, dx, gridDim);
    const int j0 = cellContaining(yp, lmin, dx, gridDim);
    const int k0 = cellContaining(zp, lmin, dx, gridDim);

    int maxStencil = maxSupportCells;
    if (maxStencil > (gridDim - 1) / 2) maxStencil = (gridDim - 1) / 2;

    int stencil = 0;
    if (hp > T(0))
    {
        stencil = static_cast<int>(ceil(T(2) * hp / dx));
        if (stencil > maxStencil) stencil = maxStencil;
    }

    // Pass 1: total the weights so they can be normalised to sum to one.
    T sumW = T(0);
    if (hp > T(0))
    {
        for (int di = -stencil; di <= stencil; ++di)
            for (int dj = -stencil; dj <= stencil; ++dj)
                for (int dk = -stencil; dk <= stencil; ++dk)
                {
                    const int i = wrapIndex(i0 + di, gridDim, periodic);
                    const int j = wrapIndex(j0 + dj, gridDim, periodic);
                    const int k = wrapIndex(k0 + dk, gridDim, periodic);
                    if (i < 0 || j < 0 || k < 0) continue;

                    const T sx = minImage(xp - cellCenterAt(i, lmin, dx), boxLength, periodic);
                    const T sy = minImage(yp - cellCenterAt(j, lmin, dx), boxLength, periodic);
                    const T sz = minImage(zp - cellCenterAt(k, lmin, dx), boxLength, periodic);
                    sumW += sphKernel(sqrt(sx * sx + sy * sy + sz * sz), hp);
                }
    }

    // Sub-cell smoothing length: the whole payload goes to the particle's own cell.
    if (!(sumW > T(0)))
    {
        const uint64_t idx = slabLocalIndex(i0, j0, k0, gridDim, numRanks);
        if (slabRank(k0, gridDim, numRanks) == rank)
        {
            atomicAdd(&grid[idx], value);
            if (needWeights) atomicAdd(&weights[idx], T(1));
        }
        else
        {
            const int pos = atomicAdd(remoteCount, 1);
            if (pos < remoteCapacity)
            {
                remoteRanks[pos]   = slabRank(k0, gridDim, numRanks);
                remoteIndices[pos] = idx;
                remoteValues[pos]  = value;
                remoteWeights[pos] = T(1);
            }
        }
        return;
    }

    // Pass 2: deposit the normalised contributions.
    const T invSum = T(1) / sumW;
    for (int di = -stencil; di <= stencil; ++di)
        for (int dj = -stencil; dj <= stencil; ++dj)
            for (int dk = -stencil; dk <= stencil; ++dk)
            {
                const int i = wrapIndex(i0 + di, gridDim, periodic);
                const int j = wrapIndex(j0 + dj, gridDim, periodic);
                const int k = wrapIndex(k0 + dk, gridDim, periodic);
                if (i < 0 || j < 0 || k < 0) continue;

                const T sx = minImage(xp - cellCenterAt(i, lmin, dx), boxLength, periodic);
                const T sy = minImage(yp - cellCenterAt(j, lmin, dx), boxLength, periodic);
                const T sz = minImage(zp - cellCenterAt(k, lmin, dx), boxLength, periodic);
                const T w  = sphKernel(sqrt(sx * sx + sy * sy + sz * sz), hp);
                if (w <= T(0)) continue;

                const T        wn  = w * invSum;
                const uint64_t idx = slabLocalIndex(i, j, k, gridDim, numRanks);
                if (slabRank(k, gridDim, numRanks) == rank)
                {
                    atomicAdd(&grid[idx], value * wn);
                    if (needWeights) atomicAdd(&weights[idx], wn);
                }
                else
                {
                    const int pos = atomicAdd(remoteCount, 1);
                    if (pos < remoteCapacity)
                    {
                        remoteRanks[pos]   = slabRank(k, gridDim, numRanks);
                        remoteIndices[pos] = idx;
                        remoteValues[pos]  = value * wn;
                        remoteWeights[pos] = wn;
                    }
                }
            }
}

// One field at a time, as in cuda_p2g: a full grid per field would not fit in device memory at
// the grid sizes this is used for.
template<typename T>
void cuda_p2g_sph(CartesianGrid<T>& grid, const std::vector<T>& x, const std::vector<T>& y,
                  const std::vector<T>& z, const std::vector<T>& h,
                  const std::vector<std::vector<T>*>& field_ptrs, int maxSupportCells, bool periodic)
{
    const size_t n         = x.size();
    const size_t localSize = grid.localSize();
    const size_t nFields   = field_ptrs.size();

    grid.ensureNumFields(nFields);

    T*       d_x = nullptr, *d_y = nullptr, *d_z = nullptr, *d_h = nullptr, *d_values = nullptr;
    T*       d_grid = nullptr, *d_weights = nullptr;
    int*     d_remoteRanks = nullptr, *d_remoteCount = nullptr;
    uint64_t* d_remoteIdx = nullptr;
    T*       d_remoteValues = nullptr, *d_remoteWeights = nullptr;

    checkCudaError(cudaMalloc(&d_grid, localSize * sizeof(T)), "d_grid");
    if (n > 0)
    {
        checkCudaError(cudaMalloc(&d_x, n * sizeof(T)), "d_x");
        checkCudaError(cudaMalloc(&d_y, n * sizeof(T)), "d_y");
        checkCudaError(cudaMalloc(&d_z, n * sizeof(T)), "d_z");
        checkCudaError(cudaMalloc(&d_h, n * sizeof(T)), "d_h");
        checkCudaError(cudaMalloc(&d_values, n * sizeof(T)), "d_values");
        checkCudaError(cudaMemcpy(d_x, x.data(), n * sizeof(T), cudaMemcpyHostToDevice), "copy x");
        checkCudaError(cudaMemcpy(d_y, y.data(), n * sizeof(T), cudaMemcpyHostToDevice), "copy y");
        checkCudaError(cudaMemcpy(d_z, z.data(), n * sizeof(T), cudaMemcpyHostToDevice), "copy z");
        checkCudaError(cudaMemcpy(d_h, h.data(), n * sizeof(T), cudaMemcpyHostToDevice), "copy h");
    }
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "d_remoteCount");

    // Only particles whose stencil crosses a slab boundary produce remote entries, so start from
    // a modest guess and grow once if the kernel reports it was too small.
    int remoteCapacity = static_cast<int>(std::min<size_t>(n * 8 + 1024, 8u << 20));

    for (size_t f = 0; f < nFields; ++f)
    {
        grid.resetSphAccumulation(f, 1);
        const bool needWeights = grid.needsCellWeights();

        if (needWeights && d_weights == nullptr)
            checkCudaError(cudaMalloc(&d_weights, localSize * sizeof(T)), "d_weights");

        if (n > 0)
            checkCudaError(cudaMemcpy(d_values, field_ptrs[f]->data(), n * sizeof(T), cudaMemcpyHostToDevice),
                           "copy values");

        std::vector<int>      h_ranks;
        std::vector<uint64_t> h_indices;
        std::vector<T>        h_values, h_weights;

        for (int attempt = 0; attempt < 2; ++attempt)
        {
            checkCudaError(cudaMemset(d_grid, 0, localSize * sizeof(T)), "zero grid");
            if (needWeights) checkCudaError(cudaMemset(d_weights, 0, localSize * sizeof(T)), "zero weights");
            const int zero = 0;
            checkCudaError(cudaMemcpy(d_remoteCount, &zero, sizeof(int), cudaMemcpyHostToDevice), "zero count");

            if (d_remoteRanks == nullptr)
            {
                checkCudaError(cudaMalloc(&d_remoteRanks, remoteCapacity * sizeof(int)), "d_remoteRanks");
                checkCudaError(cudaMalloc(&d_remoteIdx, remoteCapacity * sizeof(uint64_t)), "d_remoteIdx");
                checkCudaError(cudaMalloc(&d_remoteValues, remoteCapacity * sizeof(T)), "d_remoteValues");
                checkCudaError(cudaMalloc(&d_remoteWeights, remoteCapacity * sizeof(T)), "d_remoteWeights");
            }

            if (n > 0)
            {
                const int threads = 128;
                const int blocks  = (static_cast<int>(n) + threads - 1) / threads;
                sphDepositKernel<<<blocks, threads>>>(d_x, d_y, d_z, d_h, d_values, static_cast<int>(n),
                                                      grid.gridDim_, grid.numRanks_, grid.rank_, grid.Lmin_,
                                                      grid.Lmax_, maxSupportCells, periodic, d_grid, d_weights,
                                                      needWeights, d_remoteRanks, d_remoteIdx, d_remoteValues,
                                                      d_remoteWeights, d_remoteCount, remoteCapacity);
                checkCudaError(cudaDeviceSynchronize(), "sphDepositKernel");
            }

            int numRemote = 0;
            checkCudaError(cudaMemcpy(&numRemote, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost),
                           "remoteCount back");

            if (numRemote > remoteCapacity)
            {
                // Overflowed: the staging arrays hold only the first remoteCapacity entries, so
                // grow to the measured size and redo this field.
                remoteCapacity = numRemote + numRemote / 8 + 1024;
                cudaFree(d_remoteRanks);
                cudaFree(d_remoteIdx);
                cudaFree(d_remoteValues);
                cudaFree(d_remoteWeights);
                d_remoteRanks = nullptr;
                continue;
            }

            if (numRemote > 0)
            {
                h_ranks.resize(numRemote);
                h_indices.resize(numRemote);
                h_values.resize(numRemote);
                h_weights.resize(numRemote);
                checkCudaError(cudaMemcpy(h_ranks.data(), d_remoteRanks, numRemote * sizeof(int),
                                          cudaMemcpyDeviceToHost), "ranks back");
                checkCudaError(cudaMemcpy(h_indices.data(), d_remoteIdx, numRemote * sizeof(uint64_t),
                                          cudaMemcpyDeviceToHost), "indices back");
                checkCudaError(cudaMemcpy(h_values.data(), d_remoteValues, numRemote * sizeof(T),
                                          cudaMemcpyDeviceToHost), "values back");
                checkCudaError(cudaMemcpy(h_weights.data(), d_remoteWeights, numRemote * sizeof(T),
                                          cudaMemcpyDeviceToHost), "weights back");
            }
            break;
        }

        checkCudaError(cudaMemcpy(grid.grid_fields_[f].data(), d_grid, localSize * sizeof(T),
                                  cudaMemcpyDeviceToHost), "grid back");
        if (needWeights)
            checkCudaError(cudaMemcpy(grid.cell_weights_.data(), d_weights, localSize * sizeof(T),
                                      cudaMemcpyDeviceToHost), "weights back");

        for (size_t m = 0; m < h_ranks.size(); ++m)
        {
            const int r = h_ranks[m];
            grid.send_index_[r].push_back(h_indices[m]);
            grid.send_weights_[r].push_back(h_weights[m]);
            grid.send_values_[r][f].push_back(h_values[m]);
        }

        grid.exchangeAndReduceSph(f, 1);
    }

    cudaFree(d_grid);
    cudaFree(d_weights);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_h);
    cudaFree(d_values);
    cudaFree(d_remoteRanks);
    cudaFree(d_remoteIdx);
    cudaFree(d_remoteValues);
    cudaFree(d_remoteWeights);
    cudaFree(d_remoteCount);
}

template void cuda_p2g<double>(CartesianGrid<double>&, const std::vector<KeyType>&,
                               const std::vector<std::vector<double>*>&);

template void cuda_p2g_sph<double>(CartesianGrid<double>&, const std::vector<double>&,
                                   const std::vector<double>&, const std::vector<double>&,
                                   const std::vector<double>&, const std::vector<std::vector<double>*>&, int,
                                   bool);

} // namespace p2g
