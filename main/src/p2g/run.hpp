#pragma once

#include "p2g/config.hpp"

namespace p2g {

/** Run the full pipeline: read checkpoint → domain sync → rasterize → write density.
 *  Returns true on success, false on failure (errors printed on rank 0). */
bool run(Config const& config, int rank, int numRanks);

} // namespace p2g
