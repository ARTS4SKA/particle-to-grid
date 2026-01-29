#pragma once

#include <cstdint>
#include <vector>

namespace p2g {

using KeyType = uint64_t;

template<typename T>
struct DataSender
{
    std::vector<uint64_t>       send_index;
    std::vector<std::vector<T>> send_dens_per_field;  // send_dens_per_field[f].size() == send_index.size()
};

} // namespace p2g
