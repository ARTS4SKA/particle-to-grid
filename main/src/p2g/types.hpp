#pragma once

#include <cstdint>
#include <vector>

namespace p2g {

using KeyType = uint64_t;

template<typename T>
struct DataSender
{
    std::vector<uint64_t> send_index;
    std::vector<T>        send_dens;
};

} // namespace p2g
