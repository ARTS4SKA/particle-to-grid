/*
 * MIT License
 *
 * Copyright (c) 2023 CSCS, ETH Zurich, University of Basel, University of Zurich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief File I/O interface implementation for TIPSY format
 */

#include <mpi.h>

#include <cstring>
#include <fstream>
#include <string>
#include <variant>
#include <vector>
#include <algorithm>
#include <arpa/inet.h> // for ntohl (network byte order = big-endian)
#include <byteswap.h>  // for __bswap_64

#include "ifile_io_impl.h"

namespace sphexa
{

// Byte-swapping functions for big-endian to host conversion
inline double beToHostDouble(double val)
{
    union {
        uint64_t i;
        double   d;
    } u;
    u.d = val;
    u.i = __bswap_64(u.i); // Swap 64-bit integer bytes
    return u.d;
}

inline int beToHostInt(int val) { return static_cast<int>(ntohl(static_cast<uint32_t>(val))); }

inline float beToHostFloat(float val)
{
    union {
        uint32_t i;
        float    f;
    } u;
    u.f = val;
    u.i = ntohl(u.i);
    return u.f;
}

// TIPSY file format structures
// Note: These structures must match the binary layout exactly
// Format from read_pkdgrav.py:
// Header: 1 double (a) + 6 ints (N, Dims, Ngas, Ndark, Nstar, pad) = 8 + 24 = 32 bytes
// Dark: mass, x, y, z, vx, vy, vz, eps, phi = 9 floats = 36 bytes
// Note: mass comes FIRST, then positions, then velocities

#pragma pack(push, 1)
struct TipsyHeader
{
    double a;     // scale factor (not time)
    int    N;     // total particles
    int    Dims;  // dimensions
    int    Ngas;  // gas particles
    int    Ndark; // dark matter particles
    int    Nstar; // star particles
    int    pad;   // padding
};

struct TipsyDarkParticle
{
    float mass; // mass comes FIRST
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float eps;
    float phi;
};
#pragma pack(pop)

inline auto partitionRange(size_t R, size_t i, size_t N)
{
    size_t s = R / N;
    size_t r = R % N;
    if (i < r)
    {
        size_t start = (s + 1) * i;
        size_t end   = start + s + 1;
        return std::make_tuple(start, end);
    }
    else
    {
        size_t start = (s + 1) * r + s * (i - r);
        size_t end   = start + s;
        return std::make_tuple(start, end);
    }
}

class TipsyReader final : public IFileReader
{
public:
    using Base      = IFileReader;
    using FieldType = typename Base::FieldType;

    explicit TipsyReader(MPI_Comm comm)
        : comm_(comm)
        , isOpen_(false)
    {
        MPI_Comm_rank(comm, &rank_);
        MPI_Comm_size(comm, &numRanks_);
    }

    ~TipsyReader() override { closeStep(); }

    [[nodiscard]] int     rank() const override { return rank_; }
    [[nodiscard]] int64_t numParticles() const override
    {
        if (!isOpen_) { throw std::runtime_error("Cannot get number of particles: file not open\n"); }
        return static_cast<int64_t>(globalCount_);
    }

    void setStep(std::string path, int step, FileMode mode) override
    {
        closeStep();
        pathStep_ = path;

        // TIPSY files don't have steps, so step parameter is ignored
        (void)step;

        // Read file on rank 0 first to get header
        TipsyHeader header;
        if (rank_ == 0)
        {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) { throw std::runtime_error("Cannot open TIPSY file: " + path + "\n"); }

            // Read header (1 double + 6 ints = 8 + 24 = 32 bytes)
            file.read(reinterpret_cast<char*>(&header), sizeof(TipsyHeader));
            if (!file.good()) { throw std::runtime_error("Error reading TIPSY header\n"); }
            
            // Convert from big-endian to host byte order
            header.a     = beToHostDouble(header.a);
            header.N     = beToHostInt(header.N);
            header.Dims  = beToHostInt(header.Dims);
            header.Ngas  = beToHostInt(header.Ngas);
            header.Ndark = beToHostInt(header.Ndark);
            header.Nstar = beToHostInt(header.Nstar);
            header.pad   = beToHostInt(header.pad);
        }

        // Broadcast header to all ranks (already converted on rank 0)
        MPI_Bcast(&header, sizeof(TipsyHeader), MPI_BYTE, 0, comm_);

        // Debug output on rank 0
        if (rank_ == 0)
        {
            std::cout << "TIPSY header: a=" << header.a << ", N=" << header.N 
                      << ", Dims=" << header.Dims << ", Ngas=" << header.Ngas 
                      << ", Ndark=" << header.Ndark << ", Nstar=" << header.Nstar << std::endl;
        }

        // Only read dark matter particles (matching read_pkdgrav.py)
        globalCount_ = header.Ndark;
        if (globalCount_ < 1) 
        { 
            if (rank_ == 0)
            {
                std::cerr << "Warning: No dark matter particles found (Ndark=" << header.Ndark << ")\n";
            }
            return; 
        }

        // Calculate particle ranges for this rank (only for dark matter particles)
        if (mode == FileMode::collective)
        {
            std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank_, numRanks_);
            localCount_                       = lastIndex_ - firstIndex_;
        }
        else
        {
            std::tie(firstIndex_, lastIndex_, localCount_) = std::make_tuple(0, globalCount_, globalCount_);
        }

        // Read particles for this rank (only dark matter for now, matching read_pkdgrav.py)
        readParticles(path, header, firstIndex_, lastIndex_);

        isOpen_ = true;
    }

    std::vector<std::string> fileAttributes() override { return {}; }

    std::vector<std::string> stepAttributes() override { return {}; }

    int64_t fileAttributeSize(const std::string&) override { return 0; }

    int64_t stepAttributeSize(const std::string&) override { return 0; }

    void fileAttribute(const std::string&, FieldType, int64_t) override
    {
        throw std::runtime_error("TIPSY format does not support file attributes\n");
    }

    void stepAttribute(const std::string&, FieldType, int64_t) override
    {
        throw std::runtime_error("TIPSY format does not support step attributes\n");
    }

    void readField(const std::string& key, FieldType field) override
    {
        if (!isOpen_) { throw std::runtime_error("Cannot read field: file not open\n"); }

        if (key == "x")
        {
            std::visit([this](auto arg) { copyField(x_, arg, localCount_); }, field);
        }
        else if (key == "y")
        {
            std::visit([this](auto arg) { copyField(y_, arg, localCount_); }, field);
        }
        else if (key == "z")
        {
            std::visit([this](auto arg) { copyField(z_, arg, localCount_); }, field);
        }
        else if (key == "vx")
        {
            std::visit([this](auto arg) { copyField(vx_, arg, localCount_); }, field);
        }
        else if (key == "vy")
        {
            std::visit([this](auto arg) { copyField(vy_, arg, localCount_); }, field);
        }
        else if (key == "vz")
        {
            std::visit([this](auto arg) { copyField(vz_, arg, localCount_); }, field);
        }
        else if (key == "mass")
        {
            std::visit([this](auto arg) { copyField(mass_, arg, localCount_); }, field);
        }
        else
        {
            throw std::runtime_error("Unknown field: " + key + "\n");
        }
    }

    uint64_t localNumParticles() override { return localCount_; }

    uint64_t globalNumParticles() override { return globalCount_; }

    void closeStep() override
    {
        x_.clear();
        y_.clear();
        z_.clear();
        vx_.clear();
        vy_.clear();
        vz_.clear();
        mass_.clear();
        isOpen_ = false;
    }

private:
    template<typename T>
    void copyField(const std::vector<double>& source, T* dest, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            dest[i] = static_cast<T>(source[i]);
        }
    }

    void readParticles(const std::string& path, const TipsyHeader& header, size_t firstIndex, size_t lastIndex)
    {
        size_t localCount = lastIndex - firstIndex;
        x_.resize(localCount);
        y_.resize(localCount);
        z_.resize(localCount);
        vx_.resize(localCount);
        vy_.resize(localCount);
        vz_.resize(localCount);
        mass_.resize(localCount);

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) { throw std::runtime_error("Cannot open TIPSY file: " + path + "\n"); }

        // Skip header
        file.seekg(sizeof(TipsyHeader));

        size_t particleIndex = 0;
        size_t targetIndex   = 0;

        // Skip gas particles (read_pkdgrav.py only reads dark matter)
        for (int i = 0; i < header.Ngas; ++i)
        {
            // Gas particles not supported in this format, skip them
            // Size would be: mass, x, y, z, vx, vy, vz, rho, temp, hsmooth, metals, phi = 12 floats = 48 bytes
            file.seekg(12 * sizeof(float), std::ios::cur);
            if (!file.good()) { throw std::runtime_error("Error skipping gas particles\n"); }
        }

        // Read dark matter particles (matching read_pkdgrav.py format)
        for (int i = 0; i < header.Ndark; ++i)
        {
            TipsyDarkParticle dark;
            file.read(reinterpret_cast<char*>(&dark), sizeof(TipsyDarkParticle));
            if (!file.good()) { throw std::runtime_error("Error reading dark matter particle\n"); }

            // Convert from big-endian to host byte order
            dark.mass = beToHostFloat(dark.mass);
            dark.x    = beToHostFloat(dark.x);
            dark.y    = beToHostFloat(dark.y);
            dark.z    = beToHostFloat(dark.z);
            dark.vx   = beToHostFloat(dark.vx);
            dark.vy   = beToHostFloat(dark.vy);
            dark.vz   = beToHostFloat(dark.vz);
            dark.eps  = beToHostFloat(dark.eps);
            dark.phi  = beToHostFloat(dark.phi);

            if (particleIndex >= firstIndex && particleIndex < lastIndex)
            {
                // Format: mass, x, y, z, vx, vy, vz, eps, phi
                mass_[targetIndex] = static_cast<double>(dark.mass);
                x_[targetIndex]    = static_cast<double>(dark.x);
                y_[targetIndex]    = static_cast<double>(dark.y);
                z_[targetIndex]    = static_cast<double>(dark.z);
                vx_[targetIndex]   = static_cast<double>(dark.vx);
                vy_[targetIndex]   = static_cast<double>(dark.vy);
                vz_[targetIndex]   = static_cast<double>(dark.vz);
                targetIndex++;
            }
            particleIndex++;
        }

        // Skip star particles (read_pkdgrav.py only reads dark matter)
        for (int i = 0; i < header.Nstar; ++i)
        {
            // Star particles not supported in this format, skip them
            // Size would be: mass, x, y, z, vx, vy, vz, metals, tform, eps, phi = 11 floats = 44 bytes
            file.seekg(11 * sizeof(float), std::ios::cur);
            if (!file.good()) { throw std::runtime_error("Error skipping star particles\n"); }
        }

        if (targetIndex != localCount)
        {
            throw std::runtime_error("Particle count mismatch: expected " + std::to_string(localCount) +
                                     ", got " + std::to_string(targetIndex) + "\n");
        }
    }

    int      rank_{0};
    int      numRanks_{0};
    MPI_Comm comm_;

    uint64_t    firstIndex_{0};
    uint64_t    lastIndex_{0};
    uint64_t    localCount_{0};
    uint64_t    globalCount_{0};
    std::string pathStep_;

    std::vector<double> x_, y_, z_;
    std::vector<double> vx_, vy_, vz_;
    std::vector<double> mass_;

    bool isOpen_{false};
};

std::unique_ptr<IFileReader> makeTipsyReader(MPI_Comm comm) { return std::make_unique<TipsyReader>(comm); }

} // namespace sphexa

