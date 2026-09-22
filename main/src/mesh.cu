#include "mesh.hpp"
#include <cuda_runtime.h>
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

        grid.exchangeAndAverage(f, 1);
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

template void cuda_p2g<double>(CartesianGrid<double>&, const std::vector<KeyType>&,
                               const std::vector<std::vector<double>*>&);

} // namespace p2g
