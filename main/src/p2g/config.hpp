#pragma once

#include "p2g/interpolation_method.hpp"

#include <string>
#include <vector>

namespace p2g {

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
    bool                         write_output   = true;  // if false, skip writing density/field .txt files
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
