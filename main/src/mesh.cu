#include "mesh.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Kernel: classify particles as local or remote, and accumulate local contributions
template<class T>
__global__ void classifyAndRasterizeKernel(const KeyType* keys,
                                            const T*       mass,
                                            int            numParticles,
                                            int            gridDim,
                                            int            numRanks,
                                            int            rank,
                                            T*             dens,
                                            // Remote particle data (output)
                                            int*           remoteRanks,
                                            uint64_t*      remoteIndices,
                                            T*             remoteMass,
                                            int*           remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    KeyType key = keys[idx];

    // Same logic as Mesh::calculateKeyIndices
    auto      mesh_indices = cstone::decodeHilbert(key);
    unsigned  divisor      = 1 + static_cast<unsigned>(std::pow(2, 21)) / gridDim;
    int       i            = util::get<0>(mesh_indices) / divisor;
    int       j            = util::get<1>(mesh_indices) / divisor;
    int       k            = util::get<2>(mesh_indices) / divisor;

    if (i < 0 || i >= gridDim || j < 0 || j >= gridDim || k < 0 || k >= gridDim) return;

    // Calculate target rank (1D slab decomposition along z)
    int targetRank = k / (gridDim / numRanks);
    if (targetRank >= numRanks) targetRank = numRanks - 1;

    if (targetRank == rank)
    {
        // Local particle: calculate local index and accumulate directly
        int base = gridDim / numRanks;
        int remI = i % base;
        int remJ = j % base;
        int remK = k % base;
        uint64_t localIndex = remI + remJ * base + remK * base * base;
        
        atomicAdd(&dens[localIndex], mass[idx]);
    }
    else
    {
        // Remote particle: add to remote list
        int base = gridDim / numRanks;
        int remI = i % base;
        int remJ = j % base;
        int remK = k % base;
        uint64_t localIndex = remI + remJ * base + remK * base * base;
        
        int pos = atomicAdd(remoteCount, 1);
        remoteRanks[pos] = targetRank;
        remoteIndices[pos] = localIndex;
        remoteMass[pos] = mass[idx];
    }
}

// Kernel: accumulate received contributions into density field
template<class T>
__global__ void accumulateReceivedKernel(const uint64_t* indices,
                                         const T*        mass,
                                         int             count,
                                         T*              dens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    atomicAdd(&dens[indices[idx]], mass[idx]);
}

template<typename T>
void rasterize_particles_to_mesh_cuda(Mesh<T>&         mesh,
                                      std::vector<KeyType> keys,
                                      std::vector<T>   x,
                                      std::vector<T>   y,
                                      std::vector<T>   z,
                                      std::vector<T>   mass)
{
    std::cout << "rank " << mesh.rank_ << " rasterize start (CUDA density)" << std::endl;

    int numParticles = static_cast<int>(keys.size());
    if (numParticles == 0) return;

    int      gridDim   = mesh.gridDim_;
    int      base      = gridDim / mesh.numRanks_;
    uint64_t localSize = static_cast<uint64_t>(gridDim) * gridDim * base;

    // Ensure communication vectors are properly sized
    if (mesh.send_count.size() != static_cast<size_t>(mesh.numRanks_))
    {
        mesh.resize_comm_size(mesh.numRanks_);
    }
    std::fill(mesh.send_count.begin(), mesh.send_count.end(), 0);
    std::fill(mesh.send_disp.begin(), mesh.send_disp.end(), 0);
    std::fill(mesh.recv_count.begin(), mesh.recv_count.end(), 0);
    std::fill(mesh.recv_disp.begin(), mesh.recv_disp.end(), 0);
    
    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSender[i].send_index.clear();
        mesh.vdataSender[i].send_dens.clear();
    }

    // Reset density field
    if (mesh.dens_.size() != localSize)
    {
        mesh.dens_.assign(localSize, T(0));
    }

    // Allocate device memory
    KeyType* d_keys = nullptr;
    T*       d_mass = nullptr;
    T*       d_dens = nullptr;
    int*     d_remoteRanks = nullptr;
    uint64_t* d_remoteIndices = nullptr;
    T*       d_remoteMass = nullptr;
    int*     d_remoteCount = nullptr;

    checkCudaError(cudaMalloc(&d_keys, numParticles * sizeof(KeyType)), "Allocating d_keys");
    checkCudaError(cudaMalloc(&d_mass, numParticles * sizeof(T)), "Allocating d_mass");
    checkCudaError(cudaMalloc(&d_dens, localSize * sizeof(T)), "Allocating d_dens");
    checkCudaError(cudaMalloc(&d_remoteRanks, numParticles * sizeof(int)), "Allocating d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices, numParticles * sizeof(uint64_t)), "Allocating d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteMass, numParticles * sizeof(T)), "Allocating d_remoteMass");
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "Allocating d_remoteCount");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_keys, keys.data(), numParticles * sizeof(KeyType), cudaMemcpyHostToDevice),
                   "Copying keys to device");
    checkCudaError(cudaMemcpy(d_mass, mass.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice),
                   "Copying mass to device");
    checkCudaError(cudaMemcpy(d_dens, mesh.dens_.data(), localSize * sizeof(T), cudaMemcpyHostToDevice),
                   "Copying initial density to device");
    
    int zeroCount = 0;
    checkCudaError(cudaMemcpy(d_remoteCount, &zeroCount, sizeof(int), cudaMemcpyHostToDevice),
                   "Initializing remote count");

    // Launch classification and rasterization kernel
    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    classifyAndRasterizeKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_mass, numParticles, gridDim, mesh.numRanks_, mesh.rank_,
        d_dens, d_remoteRanks, d_remoteIndices, d_remoteMass, d_remoteCount);
    checkCudaError(cudaDeviceSynchronize(), "classifyAndRasterizeKernel execution");

    // Copy back local density and remote count
    checkCudaError(cudaMemcpy(mesh.dens_.data(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost),
                   "Copying density back to host");
    
    int h_remoteCount = 0;
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost),
                   "Copying remote count to host");

    // Process remote particles
    if (h_remoteCount > 0)
    {
        std::vector<int>      h_remoteRanks(h_remoteCount);
        std::vector<uint64_t> h_remoteIndices(h_remoteCount);
        std::vector<T>        h_remoteMass(h_remoteCount);

        checkCudaError(cudaMemcpy(h_remoteRanks.data(), d_remoteRanks, h_remoteCount * sizeof(int), cudaMemcpyDeviceToHost),
                       "Copying remote ranks to host");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices, h_remoteCount * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                       "Copying remote indices to host");
        checkCudaError(cudaMemcpy(h_remoteMass.data(), d_remoteMass, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost),
                       "Copying remote mass to host");

        // Organize remote data by rank
        for (int i = 0; i < h_remoteCount; i++)
        {
            int targetRank = h_remoteRanks[i];
            mesh.send_count[targetRank]++;
            mesh.vdataSender[targetRank].send_index.push_back(h_remoteIndices[i]);
            mesh.vdataSender[targetRank].send_dens.push_back(h_remoteMass[i]);
        }
    }

    // Free remote particle buffers
    cudaFree(d_remoteRanks);
    cudaFree(d_remoteIndices);
    cudaFree(d_remoteMass);
    cudaFree(d_remoteCount);

    // MPI Communication (same as CPU version)
    MPI_Alltoall(mesh.send_count.data(), 1, MpiType<int>{}, mesh.recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);

    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.send_disp[i + 1] = mesh.send_disp[i] + mesh.send_count[i];
        mesh.recv_disp[i + 1] = mesh.recv_disp[i] + mesh.recv_count[i];
    }

    // Prepare send buffers
    mesh.send_index.resize(mesh.send_disp[mesh.numRanks_]);
    mesh.send_dens.resize(mesh.send_disp[mesh.numRanks_]);

    for (int i = 0; i < mesh.numRanks_; i++)
    {
        for (int j = mesh.send_disp[i]; j < mesh.send_disp[i + 1]; j++)
        {
            mesh.send_index[j] = mesh.vdataSender[i].send_index[j - mesh.send_disp[i]];
            mesh.send_dens[j]  = mesh.vdataSender[i].send_dens[j - mesh.send_disp[i]];
        }
    }

    // Prepare receive buffers
    mesh.recv_index.resize(mesh.recv_disp[mesh.numRanks_]);
    mesh.recv_dens.resize(mesh.recv_disp[mesh.numRanks_]);

    MPI_Alltoallv(mesh.send_index.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<uint64_t>{}, 
                  mesh.recv_index.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
    MPI_Alltoallv(mesh.send_dens.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{}, 
                  mesh.recv_dens.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);

    // Accumulate received contributions on GPU
    if (mesh.recv_disp[mesh.numRanks_] > 0)
    {
        uint64_t* d_recvIndices = nullptr;
        T*        d_recvMass = nullptr;

        checkCudaError(cudaMalloc(&d_recvIndices, mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t)),
                       "Allocating d_recvIndices");
        checkCudaError(cudaMalloc(&d_recvMass, mesh.recv_disp[mesh.numRanks_] * sizeof(T)),
                       "Allocating d_recvMass");

        checkCudaError(cudaMemcpy(d_recvIndices, mesh.recv_index.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t), cudaMemcpyHostToDevice),
                       "Copying recv indices to device");
        checkCudaError(cudaMemcpy(d_recvMass, mesh.recv_dens.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice),
                       "Copying recv mass to device");

        // Update density on device
        checkCudaError(cudaMemcpy(d_dens, mesh.dens_.data(), localSize * sizeof(T), cudaMemcpyHostToDevice),
                       "Copying density to device for accumulation");

        int recvBlocks = (mesh.recv_disp[mesh.numRanks_] + threadsPerBlock - 1) / threadsPerBlock;
        accumulateReceivedKernel<<<recvBlocks, threadsPerBlock>>>(
            d_recvIndices, d_recvMass, mesh.recv_disp[mesh.numRanks_], d_dens);
        checkCudaError(cudaDeviceSynchronize(), "accumulateReceivedKernel execution");

        checkCudaError(cudaMemcpy(mesh.dens_.data(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost),
                       "Copying final density back to host");

        cudaFree(d_recvIndices);
        cudaFree(d_recvMass);
    }

    // Free device memory
    cudaFree(d_keys);
    cudaFree(d_mass);
    cudaFree(d_dens);

    // Clear particle buffers
    x.clear();
    y.clear();
    z.clear();
    mass.clear();
    keys.clear();

    // Clear send buffers
    for (int i = 0; i < mesh.numRanks_; i++)
    {
        mesh.vdataSender[i].send_index.clear();
        mesh.vdataSender[i].send_dens.clear();
    }

    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA density) done" << std::endl;
}

// Explicit template instantiation for double
template void rasterize_particles_to_mesh_cuda<double>(Mesh<double>&,
                                                       std::vector<KeyType>,
                                                       std::vector<double>,
                                                       std::vector<double>,
                                                       std::vector<double>,
                                                       std::vector<double>);


