#pragma once

#include "mesh.hpp"

// Hilbert key at the center of cell (ci, cj, ck); inverse of p2g::keyToCell.
inline p2g::KeyType cellToKey(int ci, int cj, int ck, int gridDim)
{
    unsigned divisor = 1u + (1u << 21) / static_cast<unsigned>(gridDim);
    unsigned px      = static_cast<unsigned>(ci) * divisor + divisor / 2;
    unsigned py      = static_cast<unsigned>(cj) * divisor + divisor / 2;
    unsigned pz      = static_cast<unsigned>(ck) * divisor + divisor / 2;
    return cstone::iHilbert<p2g::KeyType>(px, py, pz);
}
