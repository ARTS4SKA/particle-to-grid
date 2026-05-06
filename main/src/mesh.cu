#include "mesh.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cstring>

using KeyType = p2g::KeyType;

void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Z-slab decomposition: rank r owns k in [r*base, (r+1)*base). Local grid is (gridDim x gridDim x base).
// Returns the local index on the rank that owns global (i,j,k).
__device__ __host__ inline uint64_t globalToLocalIndexZSlab(int i, int j, int k, int gridDim, int numRanks)
{
    int base       = gridDim / numRanks;
    int targetRank = k / base;
    if (targetRank >= numRanks) targetRank = numRanks - 1;
    int localK     = k - targetRank * base;
    return static_cast<uint64_t>(i) + static_cast<uint64_t>(j) * gridDim + static_cast<uint64_t>(localK) * gridDim * gridDim;
}

// Helper: decode Hilbert key to grid cell indices (i,j,k)
__device__ inline void hilbertKeyToCell(KeyType key, int gridDim, int& i, int& j, int& k)
{
    auto mesh_indices = cstone::decodeHilbert(key);
    unsigned divisor  = 1u + (1u << 21) / static_cast<unsigned>(gridDim);
    i = util::get<0>(mesh_indices) / divisor;
    j = util::get<1>(mesh_indices) / divisor;
    k = util::get<2>(mesh_indices) / divisor;
}

// Helper: compute cell center from grid indices
template<class T>
__device__ inline void cellCenterDevice(int i, int j, int k, T Lmin, T dx, T& cx, T& cy, T& cz)
{
    cx = Lmin + dx * (T(i) + T(0.5));
    cy = Lmin + dx * (T(j) + T(0.5));
    cz = Lmin + dx * (T(k) + T(0.5));
}

// =====================================================================
// Nearest-Neighbor: two-pass GPU approach
// Pass 1: compute min distance per local cell via atomicMin; collect remote contributions
// Pass 2: assign value from the particle whose distance matches the minimum
// =====================================================================

template<class T>
__global__ void nnMinDistKernel(const KeyType* keys, const T* x, const T* y, const T* z, const T* mass,
                                int numParticles, int gridDim, int numRanks, int rank,
                                T Lmin, T dx,
                                unsigned long long* distBits,
                                int* remoteRanks, uint64_t* remoteIndices, T* remoteMass,
                                T* remoteDist, int* remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int i, j, k;
    hilbertKeyToCell(keys[idx], gridDim, i, j, k);
    if (i < 0 || i >= gridDim || j < 0 || j >= gridDim || k < 0 || k >= gridDim) return;

    T cx, cy, cz;
    cellCenterDevice(i, j, k, Lmin, dx, cx, cy, cz);
    T dist = sqrt((x[idx] - cx) * (x[idx] - cx) + (y[idx] - cy) * (y[idx] - cy) + (z[idx] - cz) * (z[idx] - cz));

    int base       = gridDim / numRanks;
    int targetRank = k / base;
    if (targetRank >= numRanks) targetRank = numRanks - 1;
    uint64_t localIndex = globalToLocalIndexZSlab(i, j, k, gridDim, numRanks);

    if (targetRank == rank)
    {
        unsigned long long myBits = static_cast<unsigned long long>(__double_as_longlong(dist));
        atomicMin(&distBits[localIndex], myBits);
    }
    else
    {
        int pos = atomicAdd(remoteCount, 1);
        remoteRanks[pos]   = targetRank;
        remoteIndices[pos]  = localIndex;
        remoteMass[pos]     = mass[idx];
        remoteDist[pos]     = dist;
    }
}

template<class T>
__global__ void nnAssignValueKernel(const KeyType* keys, const T* x, const T* y, const T* z, const T* mass,
                                     int numParticles, int gridDim, int numRanks, int rank,
                                     T Lmin, T dx,
                                     const unsigned long long* distBits,
                                     T* dens)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int i, j, k;
    hilbertKeyToCell(keys[idx], gridDim, i, j, k);
    if (i < 0 || i >= gridDim || j < 0 || j >= gridDim || k < 0 || k >= gridDim) return;

    int base       = gridDim / numRanks;
    int targetRank = k / base;
    if (targetRank >= numRanks) targetRank = numRanks - 1;
    if (targetRank != rank) return;

    T cx, cy, cz;
    cellCenterDevice(i, j, k, Lmin, dx, cx, cy, cz);
    T dist = sqrt((x[idx] - cx) * (x[idx] - cx) + (y[idx] - cy) * (y[idx] - cy) + (z[idx] - cz) * (z[idx] - cz));

    uint64_t localIndex = globalToLocalIndexZSlab(i, j, k, gridDim, numRanks);
    unsigned long long myBits = static_cast<unsigned long long>(__double_as_longlong(dist));
    if (myBits == distBits[localIndex])
        dens[localIndex] = mass[idx];
}

// =====================================================================
// Cell-Average: key-based, accumulate values and counts
// =====================================================================

template<class T>
__global__ void cellAverageKernel(const KeyType* keys, const T* mass,
                                   int numParticles, int gridDim, int numRanks, int rank,
                                   T* dens, int* counts,
                                   int* remoteRanks, uint64_t* remoteIndices, T* remoteMass,
                                   int* remoteCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int i, j, k;
    hilbertKeyToCell(keys[idx], gridDim, i, j, k);
    if (i < 0 || i >= gridDim || j < 0 || j >= gridDim || k < 0 || k >= gridDim) return;

    int base       = gridDim / numRanks;
    int targetRank = k / base;
    if (targetRank >= numRanks) targetRank = numRanks - 1;
    uint64_t localIndex = globalToLocalIndexZSlab(i, j, k, gridDim, numRanks);

    if (targetRank == rank)
    {
        atomicAdd(&dens[localIndex], mass[idx]);
        atomicAdd(&counts[localIndex], 1);
    }
    else
    {
        int pos = atomicAdd(remoteCount, 1);
        remoteRanks[pos]   = targetRank;
        remoteIndices[pos]  = localIndex;
        remoteMass[pos]     = mass[idx];
    }
}

// =====================================================================
// SPH kernels (unchanged)
// =====================================================================

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

constexpr int    MAX_SPH_REMOTE_PER_PARTICLE = 64;
constexpr size_t MAX_SPH_REMOTE_TOTAL        = 16 * 1024 * 1024;  // 16M entries max (~320 MB)

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
                uint64_t localIndex = globalToLocalIndexZSlab(i, j, k, gridDim, numRanks);

                if (targetRank == rank)
                    atomicAdd(&dens[localIndex], contrib);
                else
                {
                    int pos = atomicAdd(remoteCount, 1);
                    if (pos < maxRemoteEntries)
                    {
                        remoteRanks[pos]   = targetRank;
                        remoteIndices[pos]  = localIndex;
                        remoteMass[pos]    = contrib;
                    }
                }
            }
        }
    }
}

// =====================================================================
// Host wrappers
// =====================================================================

// --- Nearest Neighbor (two-pass) ---
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
    std::cout << "rank " << mesh.rank_ << " rasterize start (CUDA nearest_neighbor)" << std::endl;

    int numParticles = static_cast<int>(keys.size());
    if (numParticles == 0) return;

    int      gridDim   = mesh.gridDim_;
    uint64_t localSize = mesh.localSize();
    T        dx        = (mesh.Lmax_ - mesh.Lmin_) / static_cast<T>(gridDim);

    if (mesh.send_count.size() != static_cast<size_t>(mesh.numRanks_))
        mesh.resize_comm_size(mesh.numRanks_);
    if (doReset) mesh.resetCommAndDens();

    mesh.cell_distances_.assign(localSize, std::numeric_limits<T>::max());

    // Allocate device memory
    KeyType*           d_keys = nullptr;
    T*                 d_x = nullptr, * d_y = nullptr, * d_z = nullptr, * d_mass = nullptr, * d_dens = nullptr;
    unsigned long long* d_distBits = nullptr;
    int*               d_remoteRanks = nullptr;
    uint64_t*          d_remoteIndices = nullptr;
    T*                 d_remoteMass = nullptr;
    T*                 d_remoteDist = nullptr;
    int*               d_remoteCount = nullptr;

    checkCudaError(cudaMalloc(&d_keys, numParticles * sizeof(KeyType)), "d_keys");
    checkCudaError(cudaMalloc(&d_x, numParticles * sizeof(T)), "d_x");
    checkCudaError(cudaMalloc(&d_y, numParticles * sizeof(T)), "d_y");
    checkCudaError(cudaMalloc(&d_z, numParticles * sizeof(T)), "d_z");
    checkCudaError(cudaMalloc(&d_mass, numParticles * sizeof(T)), "d_mass");
    checkCudaError(cudaMalloc(&d_dens, localSize * sizeof(T)), "d_dens");
    checkCudaError(cudaMalloc(&d_distBits, localSize * sizeof(unsigned long long)), "d_distBits");
    checkCudaError(cudaMalloc(&d_remoteRanks, numParticles * sizeof(int)), "d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices, numParticles * sizeof(uint64_t)), "d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteMass, numParticles * sizeof(T)), "d_remoteMass");
    checkCudaError(cudaMalloc(&d_remoteDist, numParticles * sizeof(T)), "d_remoteDist");
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "d_remoteCount");

    // Copy particle data to device
    checkCudaError(cudaMemcpy(d_keys, keys.data(), numParticles * sizeof(KeyType), cudaMemcpyHostToDevice), "copy keys");
    checkCudaError(cudaMemcpy(d_x, x.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy x");
    checkCudaError(cudaMemcpy(d_y, y.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy y");
    checkCudaError(cudaMemcpy(d_z, z.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy z");
    checkCudaError(cudaMemcpy(d_mass, mass.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy mass");

    // Initialize d_dens to 0 and d_distBits to max
    checkCudaError(cudaMemset(d_dens, 0, localSize * sizeof(T)), "zero dens");
    {
        // Fill distBits with the bit pattern of max double
        double maxDist = std::numeric_limits<T>::max();
        unsigned long long maxBits;
        std::memcpy(&maxBits, &maxDist, sizeof(double));
        std::vector<unsigned long long> h_distInit(localSize, maxBits);
        checkCudaError(cudaMemcpy(d_distBits, h_distInit.data(), localSize * sizeof(unsigned long long), cudaMemcpyHostToDevice), "init distBits");
    }
    int zero = 0;
    checkCudaError(cudaMemcpy(d_remoteCount, &zero, sizeof(int), cudaMemcpyHostToDevice), "zero remoteCount");

    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;

    // Pass 1: compute min distances, collect remote contributions
    nnMinDistKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_x, d_y, d_z, d_mass, numParticles, gridDim, mesh.numRanks_, mesh.rank_,
        mesh.Lmin_, dx, d_distBits, d_remoteRanks, d_remoteIndices, d_remoteMass, d_remoteDist, d_remoteCount);
    checkCudaError(cudaDeviceSynchronize(), "nnMinDistKernel");

    // Pass 2: assign values for nearest local particles
    nnAssignValueKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_x, d_y, d_z, d_mass, numParticles, gridDim, mesh.numRanks_, mesh.rank_,
        mesh.Lmin_, dx, d_distBits, d_dens);
    checkCudaError(cudaDeviceSynchronize(), "nnAssignValueKernel");

    // Copy results back to host
    checkCudaError(cudaMemcpy(mesh.currentGridData(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost), "dens back");
    // Copy distBits → cell_distances_ (same bit representation for positive doubles)
    static_assert(sizeof(double) == sizeof(unsigned long long));
    checkCudaError(cudaMemcpy(mesh.cell_distances_.data(), d_distBits, localSize * sizeof(double), cudaMemcpyDeviceToHost), "dist back");

    int h_remoteCount = 0;
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost), "remoteCount back");

    // Organize remote contributions by rank
    if (h_remoteCount > 0)
    {
        std::vector<int>      h_remoteRanks(h_remoteCount);
        std::vector<uint64_t> h_remoteIndices(h_remoteCount);
        std::vector<T>        h_remoteMass(h_remoteCount);
        std::vector<T>        h_remoteDist(h_remoteCount);
        checkCudaError(cudaMemcpy(h_remoteRanks.data(), d_remoteRanks, h_remoteCount * sizeof(int), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices, h_remoteCount * sizeof(uint64_t), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteMass.data(), d_remoteMass, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteDist.data(), d_remoteDist, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost), "");

        for (int i = 0; i < h_remoteCount; i++)
        {
            int targetRank = h_remoteRanks[i];
            mesh.send_count[targetRank]++;
            mesh.vdataSender[targetRank].send_index.push_back(h_remoteIndices[i]);
            mesh.vdataSender[targetRank].send_dens_per_field[mesh.current_field_index_].push_back(h_remoteMass[i]);
            mesh.vdataSender[targetRank].send_distances.push_back(h_remoteDist[i]);
        }
    }

    // Free device memory
    cudaFree(d_keys); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_mass);
    cudaFree(d_dens); cudaFree(d_distBits);
    cudaFree(d_remoteRanks); cudaFree(d_remoteIndices); cudaFree(d_remoteMass);
    cudaFree(d_remoteDist); cudaFree(d_remoteCount);

    if (doExchange)
        mesh.performExchangeNearestNeighbor();

    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA nearest_neighbor) done" << std::endl;
}

// --- Cell Average (key-based) ---
template<typename T>
void rasterize_particles_to_mesh_cuda_cell_average(p2g::Mesh<T>& mesh, std::vector<KeyType> keys,
                                                   std::vector<T> mass,
                                                   bool doExchange, bool doReset)
{
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA cell_average) start" << std::endl;
    int numParticles = static_cast<int>(keys.size());
    if (numParticles == 0) return;

    int      gridDim   = mesh.gridDim_;
    uint64_t localSize = mesh.localSize();

    if (mesh.send_count.size() != static_cast<size_t>(mesh.numRanks_)) mesh.resize_comm_size(mesh.numRanks_);
    if (doReset) mesh.resetCommAndDens();
    mesh.cell_counts_.assign(localSize, 0);

    KeyType*  d_keys = nullptr;
    T*        d_mass = nullptr;
    T*        d_dens = nullptr;
    int*      d_counts = nullptr;
    int*      d_remoteRanks = nullptr;
    uint64_t* d_remoteIndices = nullptr;
    T*        d_remoteMass = nullptr;
    int*      d_remoteCount = nullptr;

    checkCudaError(cudaMalloc(&d_keys, numParticles * sizeof(KeyType)), "d_keys");
    checkCudaError(cudaMalloc(&d_mass, numParticles * sizeof(T)), "d_mass");
    checkCudaError(cudaMalloc(&d_dens, localSize * sizeof(T)), "d_dens");
    checkCudaError(cudaMalloc(&d_counts, localSize * sizeof(int)), "d_counts");
    checkCudaError(cudaMalloc(&d_remoteRanks, numParticles * sizeof(int)), "d_remoteRanks");
    checkCudaError(cudaMalloc(&d_remoteIndices, numParticles * sizeof(uint64_t)), "d_remoteIndices");
    checkCudaError(cudaMalloc(&d_remoteMass, numParticles * sizeof(T)), "d_remoteMass");
    checkCudaError(cudaMalloc(&d_remoteCount, sizeof(int)), "d_remoteCount");

    checkCudaError(cudaMemcpy(d_keys, keys.data(), numParticles * sizeof(KeyType), cudaMemcpyHostToDevice), "copy keys");
    checkCudaError(cudaMemcpy(d_mass, mass.data(), numParticles * sizeof(T), cudaMemcpyHostToDevice), "copy mass");
    checkCudaError(cudaMemset(d_dens, 0, localSize * sizeof(T)), "zero dens");
    checkCudaError(cudaMemset(d_counts, 0, localSize * sizeof(int)), "zero counts");
    int zero = 0;
    checkCudaError(cudaMemcpy(d_remoteCount, &zero, sizeof(int), cudaMemcpyHostToDevice), "zero remoteCount");

    int threadsPerBlock = 256;
    int blocksPerGrid   = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    cellAverageKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_keys, d_mass, numParticles, gridDim, mesh.numRanks_, mesh.rank_,
        d_dens, d_counts, d_remoteRanks, d_remoteIndices, d_remoteMass, d_remoteCount);
    checkCudaError(cudaDeviceSynchronize(), "cellAverageKernel");

    // Copy results back
    checkCudaError(cudaMemcpy(mesh.currentGridData(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost), "dens back");
    checkCudaError(cudaMemcpy(mesh.cell_counts_.data(), d_counts, localSize * sizeof(int), cudaMemcpyDeviceToHost), "counts back");

    int h_remoteCount = 0;
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost), "remoteCount back");

    if (h_remoteCount > 0)
    {
        std::vector<int>      h_remoteRanks(h_remoteCount);
        std::vector<uint64_t> h_remoteIndices(h_remoteCount);
        std::vector<T>        h_remoteMass(h_remoteCount);
        checkCudaError(cudaMemcpy(h_remoteRanks.data(), d_remoteRanks, h_remoteCount * sizeof(int), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteIndices.data(), d_remoteIndices, h_remoteCount * sizeof(uint64_t), cudaMemcpyDeviceToHost), "");
        checkCudaError(cudaMemcpy(h_remoteMass.data(), d_remoteMass, h_remoteCount * sizeof(T), cudaMemcpyDeviceToHost), "");

        for (int i = 0; i < h_remoteCount; i++)
        {
            int targetRank = h_remoteRanks[i];
            mesh.send_count[targetRank]++;
            mesh.vdataSender[targetRank].send_index.push_back(h_remoteIndices[i]);
            mesh.vdataSender[targetRank].send_dens_per_field[mesh.current_field_index_].push_back(h_remoteMass[i]);
        }
    }

    cudaFree(d_keys); cudaFree(d_mass); cudaFree(d_dens); cudaFree(d_counts);
    cudaFree(d_remoteRanks); cudaFree(d_remoteIndices); cudaFree(d_remoteMass); cudaFree(d_remoteCount);

    if (doExchange)
        mesh.performExchangeAndAverage();

    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA cell_average) done" << std::endl;
}

// --- SPH (unchanged logic) ---
template<typename T>
void rasterize_particles_to_mesh_cuda_sph(p2g::Mesh<T>& mesh, std::vector<T> x, std::vector<T> y, std::vector<T> z,
                                          const std::vector<T>& h, std::vector<T> mass,
                                          bool doExchange, bool doReset)
{
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA sph) start" << std::endl;
    int numParticles = static_cast<int>(x.size());
    if (numParticles == 0) return;

    int      gridDim   = mesh.gridDim_;
    uint64_t localSize = mesh.localSize();
    T        dx        = (mesh.Lmax_ - mesh.Lmin_) / static_cast<T>(gridDim);
    size_t   maxRemote = std::min(static_cast<size_t>(numParticles) * MAX_SPH_REMOTE_PER_PARTICLE, MAX_SPH_REMOTE_TOTAL);
    int      maxRemoteEntries = static_cast<int>(maxRemote);

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
        mesh.Lmin_, dx, d_dens, d_remoteRanks, d_remoteIndices, d_remoteMass, d_remoteCount, maxRemoteEntries);
    checkCudaError(cudaDeviceSynchronize(), "sph kernel");

    checkCudaError(cudaMemcpy(mesh.currentGridData(), d_dens, localSize * sizeof(T), cudaMemcpyDeviceToHost), "dens back");
    int h_remoteCount = 0;
    checkCudaError(cudaMemcpy(&h_remoteCount, d_remoteCount, sizeof(int), cudaMemcpyDeviceToHost), "remote count back");
    if (h_remoteCount > maxRemoteEntries && mesh.rank_ == 0)
        std::cerr << "SPH remote contributions overflow (got " << h_remoteCount << ", max " << maxRemoteEntries << "). Results may be incomplete." << std::endl;
    int copyCount = std::min(h_remoteCount, maxRemoteEntries);

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
            int targetRank = h_remoteRanks[i];
            if (doExchange || doReset)
            {
                mesh.send_count[targetRank]++;
                mesh.vdataSender[targetRank].send_index.push_back(h_remoteIndices[i]);
            }
            mesh.vdataSender[targetRank].send_dens_per_field[mesh.current_field_index_].push_back(h_remoteMass[i]);
        }
    }

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_h); cudaFree(d_mass);
    cudaFree(d_dens);
    cudaFree(d_remoteRanks); cudaFree(d_remoteIndices); cudaFree(d_remoteMass); cudaFree(d_remoteCount);

    if (doExchange) mesh.performExchangeAndAccumulate();
    std::cout << "rank " << mesh.rank_ << " rasterize (CUDA sph) done" << std::endl;
}

// Explicit template instantiation for double
template void rasterize_particles_to_mesh_cuda<double>(p2g::Mesh<double>&, std::vector<KeyType>,
    std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>, bool, bool);
template void rasterize_particles_to_mesh_cuda_cell_average<double>(p2g::Mesh<double>&, std::vector<KeyType>,
    std::vector<double>, bool, bool);
template void rasterize_particles_to_mesh_cuda_sph<double>(p2g::Mesh<double>&, std::vector<double>, std::vector<double>,
    std::vector<double>, const std::vector<double>&, std::vector<double>, bool, bool);
