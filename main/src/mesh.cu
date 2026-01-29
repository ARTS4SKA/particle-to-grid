#include "mesh.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

using KeyType = p2g::KeyType;

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

// Device helper: position to cell indices (clamped)
template<class T>
__device__ void positionToCellDevice(T px, T py, T pz, T Lmin, T dx, int gridDim, int& i, int& j, int& k)
{
    i = static_cast<int>((px - Lmin) / dx);
    j = static_cast<int>((py - Lmin) / dx);
    k = static_cast<int>((pz - Lmin) / dx);
    i = max(0, min(i, gridDim - 1));
    j = max(0, min(j, gridDim - 1));
    k = max(0, min(k, gridDim - 1));
}

// Kernel: cell-average rasterization (position -> cell, then same local/remote as nearest)
template<class T>
__global__ void classifyAndRasterizeCellAverageKernel(const T* x, const T* y, const T* z, const T* mass,
                                                      int numParticles, int gridDim, int numRanks, int rank,
                                                      T Lmin, T dx,
                                                      T* dens,
                                                      int* remoteRanks, uint64_t* remoteIndices, T* remoteMass,
                                                      int* remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int i, j, k;
    positionToCellDevice(x[idx], y[idx], z[idx], Lmin, dx, gridDim, i, j, k);

    int base       = gridDim / numRanks;
    int targetRank = k / base;
    if (targetRank >= numRanks) targetRank = numRanks - 1;

    int remI = i % base, remJ = j % base, remK = k % base;
    uint64_t localIndex = remI + remJ * base + remK * base * base;

    if (targetRank == rank)
        atomicAdd(&dens[localIndex], mass[idx]);
    else
    {
        int pos = atomicAdd(remoteCount, 1);
        remoteRanks[pos]  = targetRank;
        remoteIndices[pos] = localIndex;
        remoteMass[pos]   = mass[idx];
    }
}

// SPH kernel weight (3D cubic spline)
template<class T>
__device__ T sphKernelDevice(T r, T h)
{
    if (h <= T(0) || r > T(2) * h) return T(0);
    T q = r / h;
    const T sigma = T(8) / 3.14159265358979323846;
    T fac = sigma / (h * h * h);
    if (q <= T(1))
        return fac * (T(1) - T(1.5) * q * q + T(0.75) * q * q * q);
    return fac * T(0.25) * (T(2) - q) * (T(2) - q) * (T(2) - q);
}

// SPH: each particle contributes to all cells within 2h. Remote list can be large.
constexpr int MAX_SPH_REMOTE_PER_PARTICLE = 128;

template<class T>
__global__ void classifyAndRasterizeSphKernel(const T* x, const T* y, const T* z, const T* h, const T* mass,
                                              int numParticles, int gridDim, int numRanks, int rank,
                                              T Lmin, T dx,
                                              T* dens,
                                              int* remoteRanks, uint64_t* remoteIndices, T* remoteMass,
                                              int* remoteCount, int maxRemoteEntries)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    T xp = x[idx], yp = y[idx], zp = z[idx], hp = h[idx];
    if (hp <= T(0)) return;

    int i0, j0, k0;
    positionToCellDevice(xp, yp, zp, Lmin, dx, gridDim, i0, j0, k0);
    int supportCells = min(gridDim, max(1, static_cast<int>((T(2) * hp / dx)) + 1));
    int base = gridDim / numRanks;

    for (int di = -supportCells; di <= supportCells; di++)
    {
        for (int dj = -supportCells; dj <= supportCells; dj++)
        {
            for (int dk = -supportCells; dk <= supportCells; dk++)
            {
                int i = i0 + di, j = j0 + dj, k = k0 + dk;
                if (i < 0 || i >= gridDim || j < 0 || j >= gridDim || k < 0 || k >= gridDim) continue;

                T cx = Lmin + dx * (T(i) + T(0.5));
                T cy = Lmin + dx * (T(j) + T(0.5));
                T cz = Lmin + dx * (T(k) + T(0.5));
                T r  = sqrt((xp - cx) * (xp - cx) + (yp - cy) * (yp - cy) + (zp - cz) * (zp - cz));
                T w  = sphKernelDevice(r, hp);
                if (w <= T(0)) continue;

                T contrib = mass[idx] * w;
                int targetRank = k / base;
                if (targetRank >= numRanks) targetRank = numRanks - 1;
                int remI = i % base, remJ = j % base, remK = k % base;
                uint64_t localIndex = remI + remJ * base + remK * base * base;

                if (targetRank == rank)
                    atomicAdd(&dens[localIndex], contrib);
                else
                {
                    int pos = atomicAdd(remoteCount, 1);
                    if (pos < maxRemoteEntries)
                    {
                        remoteRanks[pos]   = targetRank;
                        remoteIndices[pos] = localIndex;
                        remoteMass[pos]    = contrib;
                    }
                }
            }
        }
    }
}

template<typename T>
void rasterize_particles_to_mesh_cuda(p2g::Mesh<T>&   mesh,
                                      std::vector<KeyType> keys,
                                      std::vector<T>   x,
                                      std::vector<T>   y,
                                      std::vector<T>   z,
                                      std::vector<T>   mass,
                                      bool doExchange,
                                      bool doReset)
{
    std::cout << "rank " << mesh.rank_ << " rasterize start (CUDA density)" << std::endl;

    int numParticles = static_cast<int>(keys.size());
    if (numParticles == 0) return;

    int      gridDim   = mesh.gridDim_;
    int      base      = gridDim / mesh.numRanks_;
    uint64_t localSize = static_cast<uint64_t>(gridDim) * gridDim * base;

    if (mesh.send_count.size() != static_cast<size_t>(mesh.numRanks_))
        mesh.resize_comm_size(mesh.numRanks_);

    if (doReset)
        mesh.resetCommAndDens();
    else
    {
        if (mesh.currentGridSize() != localSize)
            mesh.currentGrid().assign(localSize, T(0));
        else
            std::fill(mesh.currentGrid().begin(), mesh.currentGrid().end(), T(0));
        size_t fi = mesh.current_field_index_;
        for (int i = 0; i < mesh.numRanks_; i++)
            if (fi < mesh.vdataSender[i].send_dens_per_field.size())
                mesh.vdataSender[i].send_dens_per_field[fi].clear();
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
    checkCudaError(cudaMemcpy(d_dens, mesh.currentGridData(), localSize * sizeof(T), cudaMemcpyHostToDevice),
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
    checkCudaError(cudaMemcpy(mesh.currentGridData(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost),
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
            mesh.vdataSender[targetRank].send_dens_per_field[mesh.current_field_index_].push_back(h_remoteMass[i]);
        }
    }

    // Free remote particle buffers
    cudaFree(d_remoteRanks);
    cudaFree(d_remoteIndices);
    cudaFree(d_remoteMass);
    cudaFree(d_remoteCount);

    if (doExchange)
    {
        MPI_Alltoall(mesh.send_count.data(), 1, MpiType<int>{}, mesh.recv_count.data(), 1, MpiType<int>{}, MPI_COMM_WORLD);
        for (int i = 0; i < mesh.numRanks_; i++)
        {
            mesh.send_disp[i + 1] = mesh.send_disp[i] + mesh.send_count[i];
            mesh.recv_disp[i + 1] = mesh.recv_disp[i] + mesh.recv_count[i];
        }
        mesh.send_index.resize(mesh.send_disp[mesh.numRanks_]);
        mesh.send_dens.resize(mesh.send_disp[mesh.numRanks_]);
        size_t fi = mesh.current_field_index_;
        for (int i = 0; i < mesh.numRanks_; i++)
            for (size_t j = mesh.send_disp[i]; j < mesh.send_disp[i + 1]; j++)
            {
                mesh.send_index[j] = mesh.vdataSender[i].send_index[j - mesh.send_disp[i]];
                mesh.send_dens[j]  = mesh.vdataSender[i].send_dens_per_field[fi][j - mesh.send_disp[i]];
            }
        mesh.recv_index.resize(mesh.recv_disp[mesh.numRanks_]);
        mesh.recv_dens.resize(mesh.recv_disp[mesh.numRanks_]);
        MPI_Alltoallv(mesh.send_index.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<uint64_t>{},
                      mesh.recv_index.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<uint64_t>{}, MPI_COMM_WORLD);
        MPI_Alltoallv(mesh.send_dens.data(), mesh.send_count.data(), mesh.send_disp.data(), MpiType<T>{},
                      mesh.recv_dens.data(), mesh.recv_count.data(), mesh.recv_disp.data(), MpiType<T>{}, MPI_COMM_WORLD);
        if (mesh.recv_disp[mesh.numRanks_] > 0)
        {
            uint64_t* d_recvIndices = nullptr;
            T*        d_recvMass = nullptr;
            checkCudaError(cudaMalloc(&d_recvIndices, mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t)), "d_recvIndices");
            checkCudaError(cudaMalloc(&d_recvMass, mesh.recv_disp[mesh.numRanks_] * sizeof(T)), "d_recvMass");
            checkCudaError(cudaMemcpy(d_recvIndices, mesh.recv_index.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(uint64_t), cudaMemcpyHostToDevice), "");
            checkCudaError(cudaMemcpy(d_recvMass, mesh.recv_dens.data(), mesh.recv_disp[mesh.numRanks_] * sizeof(T), cudaMemcpyHostToDevice), "");
            checkCudaError(cudaMemcpy(d_dens, mesh.currentGridData(), localSize * sizeof(T), cudaMemcpyHostToDevice), "");
            int recvBlocks = (mesh.recv_disp[mesh.numRanks_] + threadsPerBlock - 1) / threadsPerBlock;
            accumulateReceivedKernel<<<recvBlocks, threadsPerBlock>>>(d_recvIndices, d_recvMass, mesh.recv_disp[mesh.numRanks_], d_dens);
            checkCudaError(cudaDeviceSynchronize(), "accumulateReceivedKernel");
            checkCudaError(cudaMemcpy(mesh.currentGridData(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost), "");
            cudaFree(d_recvIndices);
            cudaFree(d_recvMass);
        }
        mesh.convertMassToDensity();
        for (int i = 0; i < mesh.numRanks_; i++)
        {
            mesh.vdataSender[i].send_index.clear();
            for (auto& v : mesh.vdataSender[i].send_dens_per_field)
                v.clear();
        }
    }

    cudaFree(d_keys);
    cudaFree(d_mass);
    cudaFree(d_dens);
    if (doExchange)
    {
        x.clear();
        y.clear();
        z.clear();
        mass.clear();
        keys.clear();
    }
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA nearest) done" << std::endl;
}

template<typename T>
void rasterize_particles_to_mesh_cuda_cell_average(p2g::Mesh<T>& mesh, std::vector<T> x, std::vector<T> y,
                                                   std::vector<T> z, std::vector<T> mass,
                                                   bool doExchange, bool doReset)
{
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA cell_average) start" << std::endl;
    int numParticles = static_cast<int>(x.size());
    if (numParticles == 0) return;

    int      gridDim   = mesh.gridDim_;
    int      base      = gridDim / mesh.numRanks_;
    uint64_t localSize = static_cast<uint64_t>(gridDim) * gridDim * base;
    T        dx        = (mesh.Lmax_ - mesh.Lmin_) / static_cast<T>(gridDim);

    if (mesh.send_count.size() != static_cast<size_t>(mesh.numRanks_)) mesh.resize_comm_size(mesh.numRanks_);
    if (doReset) mesh.resetCommAndDens();
    else
    {
        mesh.currentGrid().assign(localSize, T(0));
        size_t fi = mesh.current_field_index_;
        for (int i = 0; i < mesh.numRanks_; i++)
            if (fi < mesh.vdataSender[i].send_dens_per_field.size())
                mesh.vdataSender[i].send_dens_per_field[fi].clear();
    }

    T* d_x = nullptr, * d_y = nullptr, * d_z = nullptr, * d_mass = nullptr, * d_dens = nullptr;
    int* d_remoteRanks = nullptr;
    uint64_t* d_remoteIndices = nullptr;
    T* d_remoteMass = nullptr;
    int* d_remoteCount = nullptr;

    checkCudaError(cudaMalloc(&d_x, numParticles * sizeof(T)), "d_x");
    checkCudaError(cudaMalloc(&d_y, numParticles * sizeof(T)), "d_y");
    checkCudaError(cudaMalloc(&d_z, numParticles * sizeof(T)), "d_z");
    checkCudaError(cudaMalloc(&d_mass, numParticles * sizeof(T)), "d_mass");
    checkCudaError(cudaMalloc(&d_dens, localSize * sizeof(T)), "d_dens");
    checkCudaError(cudaMalloc(&d_remoteRanks, numParticles * sizeof(int)), "d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices, numParticles * sizeof(uint64_t)), "d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteMass, numParticles * sizeof(T)), "d_remoteMass");
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "d_remoteCount");

    checkCudaError(cudaMemcpy(d_x, x.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy x");
    checkCudaError(cudaMemcpy(d_y, y.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy y");
    checkCudaError(cudaMemcpy(d_z, z.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy z");
    checkCudaError(cudaMemcpy(d_mass, mass.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy mass");
    checkCudaError(cudaMemcpy(d_dens, mesh.currentGridData(), localSize * sizeof(T), cudaMemcpyHostToDevice), "copy dens");
    int zero = 0;
    checkCudaError(cudaMemcpy(d_remoteCount, &zero, sizeof(int), cudaMemcpyHostToDevice), "remote count");

    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    classifyAndRasterizeCellAverageKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_z, d_mass, numParticles, gridDim, mesh.numRanks_, mesh.rank_,
        mesh.Lmin_, dx, d_dens, d_remoteRanks, d_remoteIndices, d_remoteMass, d_remoteCount);
    checkCudaError(cudaDeviceSynchronize(), "cell average kernel");

    checkCudaError(cudaMemcpy(mesh.currentGridData(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost), "dens back");
    int h_remoteCount = 0;
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost), "remote count back");

    if (h_remoteCount > 0)
    {
        std::vector<int> h_remoteRanks(h_remoteCount);
        std::vector<uint64_t> h_remoteIndices(h_remoteCount);
        std::vector<T> h_remoteMass(h_remoteCount);
        checkCudaError(cudaMemcpy(h_remoteRanks.data(), d_remoteRanks, h_remoteCount * sizeof(int), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices, h_remoteCount * sizeof(uint64_t), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteMass.data(), d_remoteMass, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost), "");
        for (int i = 0; i < h_remoteCount; i++)
        {
            mesh.send_count[h_remoteRanks[i]]++;
            mesh.vdataSender[h_remoteRanks[i]].send_index.push_back(h_remoteIndices[i]);
            mesh.vdataSender[h_remoteRanks[i]].send_dens_per_field[mesh.current_field_index_].push_back(h_remoteMass[i]);
        }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_mass); cudaFree(d_remoteRanks);
    cudaFree(d_remoteIndices); cudaFree(d_remoteMass); cudaFree(d_remoteCount);

    if (doExchange)
    {
        mesh.performExchangeAndAccumulate();
        mesh.convertMassToDensity();
    }
    cudaFree(d_dens);
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA cell_average) done" << std::endl;
}

template<typename T>
void rasterize_particles_to_mesh_cuda_sph(p2g::Mesh<T>& mesh, std::vector<T> x, std::vector<T> y, std::vector<T> z,
                                          const std::vector<T>& h, std::vector<T> mass,
                                          bool doExchange, bool doReset)
{
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA sph) start" << std::endl;
    int numParticles = static_cast<int>(x.size());
    if (numParticles == 0) return;

    int      gridDim   = mesh.gridDim_;
    int      base      = gridDim / mesh.numRanks_;
    uint64_t localSize = static_cast<uint64_t>(gridDim) * gridDim * base;
    T        dx        = (mesh.Lmax_ - mesh.Lmin_) / static_cast<T>(gridDim);
    int      maxRemote = numParticles * MAX_SPH_REMOTE_PER_PARTICLE;

    if (mesh.send_count.size() != static_cast<size_t>(mesh.numRanks_)) mesh.resize_comm_size(mesh.numRanks_);
    if (doReset) mesh.resetCommAndDens();
    else
    {
        mesh.currentGrid().assign(localSize, T(0));
        size_t fi = mesh.current_field_index_;
        for (int i = 0; i < mesh.numRanks_; i++)
            if (fi < mesh.vdataSender[i].send_dens_per_field.size())
                mesh.vdataSender[i].send_dens_per_field[fi].clear();
    }

    T* d_x = nullptr, * d_y = nullptr, * d_z = nullptr, * d_h = nullptr, * d_mass = nullptr, * d_dens = nullptr;
    int* d_remoteRanks = nullptr;
    uint64_t* d_remoteIndices = nullptr;
    T* d_remoteMass = nullptr;
    int* d_remoteCount = nullptr;

    checkCudaError(cudaMalloc(&d_x, numParticles * sizeof(T)), "d_x");
    checkCudaError(cudaMalloc(&d_y, numParticles * sizeof(T)), "d_y");
    checkCudaError(cudaMalloc(&d_z, numParticles * sizeof(T)), "d_z");
    checkCudaError(cudaMalloc(&d_h, numParticles * sizeof(T)), "d_h");
    checkCudaError(cudaMalloc(&d_mass, numParticles * sizeof(T)), "d_mass");
    checkCudaError(cudaMalloc(&d_dens, localSize * sizeof(T)), "d_dens");
    checkCudaError(cudaMalloc(&d_remoteRanks, maxRemote * sizeof(int)), "d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices, maxRemote * sizeof(uint64_t)), "d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteMass, maxRemote * sizeof(T)), "d_remoteMass");
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "d_remoteCount");

    checkCudaError(cudaMemcpy(d_x, x.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy x");
    checkCudaError(cudaMemcpy(d_y, y.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy y");
    checkCudaError(cudaMemcpy(d_z, z.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy z");
    checkCudaError(cudaMemcpy(d_h, h.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy h");
    checkCudaError(cudaMemcpy(d_mass, mass.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy mass");
    checkCudaError(cudaMemcpy(d_dens, mesh.currentGridData(), localSize * sizeof(T), cudaMemcpyHostToDevice), "copy dens");
    int zero = 0;
    checkCudaError(cudaMemcpy(d_remoteCount, &zero, sizeof(int), cudaMemcpyHostToDevice), "remote count");

    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    classifyAndRasterizeSphKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_x, d_y, d_z, d_h, d_mass, numParticles, gridDim, mesh.numRanks_, mesh.rank_,
        mesh.Lmin_, dx, d_dens, d_remoteRanks, d_remoteIndices, d_remoteMass, d_remoteCount, maxRemote);
    checkCudaError(cudaDeviceSynchronize(), "sph kernel");

    checkCudaError(cudaMemcpy(mesh.currentGridData(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost), "dens back");
    int h_remoteCount = 0;
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost), "remote count back");
    if (h_remoteCount > maxRemote && mesh.rank_ == 0)
        std::cerr << "SPH remote contributions overflow (got " << h_remoteCount << ", max " << maxRemote << "). Results may be incomplete." << std::endl;
    int copyCount = std::min(h_remoteCount, maxRemote);

    if (copyCount > 0)
    {
        std::vector<int> h_remoteRanks(copyCount);
        std::vector<uint64_t> h_remoteIndices(copyCount);
        std::vector<T> h_remoteMass(copyCount);
        checkCudaError(cudaMemcpy(h_remoteRanks.data(), d_remoteRanks, copyCount * sizeof(int), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices, copyCount * sizeof(uint64_t), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteMass.data(), d_remoteMass, copyCount * sizeof(T), cudaMemcpyDeviceToHost), "");
        for (int i = 0; i < copyCount; i++)
        {
            mesh.send_count[h_remoteRanks[i]]++;
            mesh.vdataSender[h_remoteRanks[i]].send_index.push_back(h_remoteIndices[i]);
            mesh.vdataSender[h_remoteRanks[i]].send_dens_per_field[mesh.current_field_index_].push_back(h_remoteMass[i]);
        }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_h); cudaFree(d_mass);
    cudaFree(d_remoteRanks); cudaFree(d_remoteIndices); cudaFree(d_remoteMass); cudaFree(d_remoteCount);

    if (doExchange) mesh.performExchangeAndAccumulate();
    cudaFree(d_dens);
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA sph) done" << std::endl;
}

// Explicit template instantiation for double
template void rasterize_particles_to_mesh_cuda<double>(p2g::Mesh<double>&, std::vector<KeyType>,
    std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, bool, bool);
template void rasterize_particles_to_mesh_cuda_cell_average<double>(p2g::Mesh<double>&, std::vector<double>,
    std::vector<double>, std::vector<double>, std::vector<double>, bool, bool);
template void rasterize_particles_to_mesh_cuda_sph<double>(p2g::Mesh<double>&, std::vector<double>, std::vector<double>,
    std::vector<double>, const std::vector<double>&, std::vector<double>, bool, bool);


