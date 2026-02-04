#pragma once

#include "p2g/interpolation_method.hpp"

#include <string>
#include <vector>

namespace p2g {

enum class OutputFormat { Text, HDF5 };

inline OutputFormat parseOutputFormat(const std::string& s)
{
    std::string lower = s;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "hdf5" || lower == "h5") return OutputFormat::HDF5;
    if (lower == "text" || lower == "txt") return OutputFormat::Text;
    throw std::runtime_error("Unknown output format '" + s + "'. Use 'text' or 'hdf5'.");
}

inline std::string to_string(OutputFormat fmt)
{
    switch (fmt)
    {
        case OutputFormat::Text: return "text";
        case OutputFormat::HDF5: return "hdf5";
    }
    return "unknown";
}

struct Config
{
    std::string                  checkpoint_path;
    std::string                  checkpoint_type = "hdf5";
    int                          step_no         = 0;
    int                          grid_size       = 0;
    double                       lbox            = 0.0;
    double                       rho_crit        = 0.0;
    InterpolationMethod          interpolation   = InterpolationMethod::NearestNeighbor;
    std::vector<std::string>     extra_field_names;  // e.g. {"temp", "vx"} for additional quantities to rasterize
    bool                         write_output    = true;   // if false, skip writing output files
    OutputFormat                 output_format   = OutputFormat::Text;
    std::string                  output_path     = "density";  // base name for output (without extension)
};

/** Validates config. On failure, sets errMsg (on rank 0) and returns false. */
inline bool validate(Config const& c, int rank, std::string& errMsg)
{
    errMsg.clear();
    if (c.checkpoint_path.empty())
    {
        errMsg = "Missing --checkpoint <path>.";
        return false;
    }
    std::string typeLower = c.checkpoint_type;
    for (char& ch : typeLower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (typeLower != "hdf5" && typeLower != "tipsy")
    {
        errMsg = "Invalid --checkpoint-type '" + c.checkpoint_type + "'. Use 'hdf5' or 'tipsy'.";
        return false;
    }
    if (typeLower == "tipsy" && c.lbox <= 0.0)
    {
        errMsg = "TIPSY requires --lbox > 0 (box size in length units).";
        return false;
    }
    if (c.grid_size < 0)
    {
        errMsg = "--gridSize must be non-negative.";
        return false;
    }
    return true;
}

} // namespace p2g
