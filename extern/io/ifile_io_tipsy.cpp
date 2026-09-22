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
#include <iostream>
#include <stdexcept>
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

    void setStep(std::string path, int /*step*/, FileMode mode) override
    {
        closeStep();
        pathStep_ = path;

        // Rank 0 validates the file and broadcasts the outcome, so that all ranks fail together
        // instead of the others blocking in a collective call.
        TipsyHeader header{};
        std::string error;
        if (rank_ == 0) { error = readHeader(path, header); }
        throwOnError(broadcastString(error));
        MPI_Bcast(&header, sizeof(TipsyHeader), MPI_BYTE, 0, comm_);

        if (rank_ == 0)
        {
            std::cout << "TIPSY header: a=" << header.a << ", N=" << header.N << ", Dims=" << header.Dims
                      << ", Ngas=" << header.Ngas << ", Ndark=" << header.Ndark << ", Nstar=" << header.Nstar
                      << std::endl;
        }

        // Only dark matter particles are read (matching read_pkdgrav.py)
        globalCount_ = header.Ndark;
        if (globalCount_ < 1)
        {
            if (rank_ == 0) { std::cerr << "Warning: No dark matter particles found (Ndark=" << header.Ndark << ")\n"; }
            return;
        }

        if (mode == FileMode::collective)
        {
            std::tie(firstIndex_, lastIndex_) = partitionRange(globalCount_, rank_, numRanks_);
        }
        else
        {
            std::tie(firstIndex_, lastIndex_) = std::make_tuple(0, globalCount_);
        }
        localCount_ = lastIndex_ - firstIndex_;

        std::string localError;
        try
        {
            readParticles(path, header, firstIndex_, lastIndex_);
        }
        catch (const std::exception& e)
        {
            localError = e.what();
        }
        int anyFailed = localError.empty() ? 0 : 1;
        MPI_Allreduce(MPI_IN_PLACE, &anyFailed, 1, MPI_INT, MPI_MAX, comm_);
        if (anyFailed)
        {
            closeStep();
            throw std::runtime_error(localError.empty() ? "Reading TIPSY particles failed on another rank\n"
                                                        : localError);
        }

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

        // "m" and "h" follow the HDF5 naming used by the pipeline and tipsy_to_hdf5.py.
        const std::vector<double>* source = nullptr;
        if (key == "x") { source = &x_; }
        else if (key == "y") { source = &y_; }
        else if (key == "z") { source = &z_; }
        else if (key == "vx") { source = &vx_; }
        else if (key == "vy") { source = &vy_; }
        else if (key == "vz") { source = &vz_; }
        else if (key == "m" || key == "mass") { source = &mass_; }
        else if (key == "h" || key == "eps") { source = &eps_; }
        else if (key == "phi") { source = &phi_; }
        else { throw std::runtime_error("Unknown field: " + key + "\n"); }

        std::visit([this, source](auto arg) { copyField(*source, arg, localCount_); }, field);
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
        eps_.clear();
        phi_.clear();
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

    // Returns an empty string on success, otherwise the error message.
    static std::string readHeader(const std::string& path, TipsyHeader& header)
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) { return "Cannot open TIPSY file: " + path + "\n"; }
        const uint64_t fileSize = file.tellg();

        file.seekg(0);
        file.read(reinterpret_cast<char*>(&header), sizeof(TipsyHeader));
        if (!file.good()) { return "Error reading TIPSY header: " + path + "\n"; }

        header.a     = beToHostDouble(header.a);
        header.N     = beToHostInt(header.N);
        header.Dims  = beToHostInt(header.Dims);
        header.Ngas  = beToHostInt(header.Ngas);
        header.Ndark = beToHostInt(header.Ndark);
        header.Nstar = beToHostInt(header.Nstar);
        header.pad   = beToHostInt(header.pad);

        if (header.Ngas < 0 || header.Ndark < 0 || header.Nstar < 0)
        {
            return "Invalid TIPSY header (negative particle count): " + path + "\n";
        }
        const uint64_t expected = sizeof(TipsyHeader) + uint64_t(header.Ngas) * gasRecordSize +
                                  uint64_t(header.Ndark) * sizeof(TipsyDarkParticle) +
                                  uint64_t(header.Nstar) * starRecordSize;
        if (fileSize < expected)
        {
            return "TIPSY file is truncated: " + path + " has " + std::to_string(fileSize) + " bytes, header implies " +
                   std::to_string(expected) + "\n";
        }
        return {};
    }

    // Broadcasts rank 0's string to all ranks and returns it.
    std::string broadcastString(std::string s) const
    {
        int len = static_cast<int>(s.size());
        MPI_Bcast(&len, 1, MPI_INT, 0, comm_);
        s.resize(len);
        MPI_Bcast(s.data(), len, MPI_CHAR, 0, comm_);
        return s;
    }

    static void throwOnError(const std::string& error)
    {
        if (!error.empty()) { throw std::runtime_error(error); }
    }

    // Reads dark particles [firstIndex, lastIndex); gas records precede them in the file.
    void readParticles(const std::string& path, const TipsyHeader& header, size_t firstIndex, size_t lastIndex)
    {
        const size_t localCount = lastIndex - firstIndex;

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) { throw std::runtime_error("Cannot open TIPSY file: " + path + "\n"); }

        const uint64_t offset =
            sizeof(TipsyHeader) + uint64_t(header.Ngas) * gasRecordSize + firstIndex * sizeof(TipsyDarkParticle);
        std::vector<TipsyDarkParticle> records(localCount);
        file.seekg(offset);
        file.read(reinterpret_cast<char*>(records.data()), localCount * sizeof(TipsyDarkParticle));
        if (!file.good()) { throw std::runtime_error("Error reading dark matter particles from " + path + "\n"); }

        for (auto* v : {&mass_, &x_, &y_, &z_, &vx_, &vy_, &vz_, &eps_, &phi_})
        {
            v->resize(localCount);
        }
        for (size_t i = 0; i < localCount; ++i)
        {
            const auto& d = records[i];
            mass_[i]      = beToHostFloat(d.mass);
            x_[i]         = beToHostFloat(d.x);
            y_[i]         = beToHostFloat(d.y);
            z_[i]         = beToHostFloat(d.z);
            vx_[i]        = beToHostFloat(d.vx);
            vy_[i]        = beToHostFloat(d.vy);
            vz_[i]        = beToHostFloat(d.vz);
            eps_[i]       = beToHostFloat(d.eps);
            phi_[i]       = beToHostFloat(d.phi);
        }
    }

    // Gas: mass, x, y, z, vx, vy, vz, rho, temp, hsmooth, metals, phi
    static constexpr uint64_t gasRecordSize = 12 * sizeof(float);
    // Star: mass, x, y, z, vx, vy, vz, metals, tform, eps, phi
    static constexpr uint64_t starRecordSize = 11 * sizeof(float);

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
    std::vector<double> mass_, eps_, phi_;

    bool isOpen_{false};
};

std::unique_ptr<IFileReader> makeTipsyReader(MPI_Comm comm) { return std::make_unique<TipsyReader>(comm); }

} // namespace sphexa

