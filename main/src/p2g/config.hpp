#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace p2g {

enum class CheckpointType { HDF5, Tipsy };
enum class OutputFormat { Text, HDF5 };

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
}

} // namespace p2g
