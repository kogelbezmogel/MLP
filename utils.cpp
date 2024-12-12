#include "utils.h"


std::string str_representation( std::vector<size_t> size_vec ) {
    std::string representation = "[ ";
    for(size_t el : size_vec)
        representation += std::to_string(el) + " ";
    representation += "]";
    return representation;
}

