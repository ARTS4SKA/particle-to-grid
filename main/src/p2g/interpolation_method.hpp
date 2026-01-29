#pragma once

#include <cctype>
#include <stdexcept>
#include <string>

namespace p2g {

enum class InterpolationMethod
{
    NearestNeighbor,
    SPH,
    CellAverage
};

inline std::string to_string(InterpolationMethod m)
{
    switch (m)
    {
        case InterpolationMethod::NearestNeighbor: return "nearest";
        case InterpolationMethod::SPH: return "sph";
        case InterpolationMethod::CellAverage: return "cell_average";
    }
    return "unknown";
}

inline InterpolationMethod parseInterpolationMethod(const std::string& s)
{
    std::string lower;
    lower.reserve(s.size());
    for (char c : s) lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (lower == "nearest" || lower == "nearest_neighbor" || lower == "ngp") return InterpolationMethod::NearestNeighbor;
    if (lower == "sph") return InterpolationMethod::SPH;
    if (lower == "cell_average" || lower == "cell-average" || lower == "average") return InterpolationMethod::CellAverage;

    throw std::invalid_argument("Unknown interpolation method: " + s +
                                ". Use: nearest, sph, or cell_average.");
}

} // namespace p2g
