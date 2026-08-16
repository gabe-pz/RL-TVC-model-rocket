#pragma once



#include <array>
#include "../rl/mlp.h" 

std::array<float, 2> policy(const std::array<double, 4>& s, const std::array<std::array<float, 4>, 64>& w1, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 64>, 4>& w3, const std::array<float, 64>& b1, 
    const std::array<float, 64>& b2, const std::array<float, 4>& b3);