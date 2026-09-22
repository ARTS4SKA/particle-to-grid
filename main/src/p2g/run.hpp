#pragma once

#include "p2g/config.hpp"

namespace p2g {

// Read checkpoint -> domain sync -> P2G -> write grid fields. Throws on failure.
void run(const Config& config, int rank, int numRanks);

} // namespace p2g
