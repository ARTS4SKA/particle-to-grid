#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace p2g {

enum class CheckpointType { HDF5, Tipsy };
enum class OutputFormat { Text, HDF5 };

// How particles are mapped onto cells.
//   NearestCell — a particle lands entirely in the cell containing it.
//   Sph         — a particle is spread over the cells within its kernel support (2h), which
//                 needs a smoothing length per particle.
enum class Interpolation { NearestCell, Sph };

inline std::string toLower(std::string s)
{
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline CheckpointType parseCheckpointType(const std::string& s)
{
    const std::string lower = toLower(s);
    if (lower == "hdf5") return CheckpointType::HDF5;
    if (lower == "tipsy") return CheckpointType::Tipsy;
    throw std::invalid_argument("Invalid --checkpoint-type '" + s + "'. Use 'hdf5' or 'tipsy'.");
}

inline Interpolation parseInterpolation(const std::string& s)
{
    const std::string lower = toLower(s);
    if (lower == "ngp" || lower == "nearest" || lower == "nearest-cell") return Interpolation::NearestCell;
    if (lower == "sph") return Interpolation::Sph;
    throw std::invalid_argument("Unknown interpolation '" + s + "'. Use 'ngp' or 'sph'.");
}

inline OutputFormat parseOutputFormat(const std::string& s)
{
    const std::string lower = toLower(s);
    if (lower == "hdf5" || lower == "h5") return OutputFormat::HDF5;
    if (lower == "text" || lower == "txt") return OutputFormat::Text;
    throw std::invalid_argument("Unknown output format '" + s + "'. Use 'text' or 'hdf5'.");
}

struct Config
{
    std::string    checkpoint_path;
    CheckpointType checkpoint_type = CheckpointType::HDF5;
    int            step_no         = 0;
    int            grid_size       = 0;
    double         lbox            = 0.0;
    double         rho_crit        = 0.0;
    Interpolation interpolation = Interpolation::NearestCell;
    int           max_support_cells = 8; // caps the SPH stencil half-width, in cells
    // Mass is always rasterized as field 0; these are the additional fields.
    std::vector<std::string> extra_field_names;
    bool                     write_output  = true;
    OutputFormat             output_format = OutputFormat::Text;
    std::string              output_path   = "density";
};

inline void validate(const Config& c)
{
    if (c.checkpoint_path.empty()) throw std::invalid_argument("Missing --checkpoint <path>.");
    if (c.checkpoint_type == CheckpointType::Tipsy && c.lbox <= 0.0)
        throw std::invalid_argument("TIPSY requires --lbox > 0 (box size in length units).");
    if (c.grid_size < 0) throw std::invalid_argument("--gridSize must be non-negative.");
    if (c.max_support_cells < 1) throw std::invalid_argument("--maxSupportCells must be at least 1.");
}

} // namespace p2g
