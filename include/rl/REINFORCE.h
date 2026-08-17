#pragma once

#include <array>
#include <algorithm>


std::array<float, 4740> gradientTermREINFORCE(const std::array<float, 4740>& gradX, const std::array<float, 4740>& gradY, float alpha, float G);

std::array<float, 4740> flattenParameters(const std::array<std::array<float, 4>, 64>& w1, const std::array<float, 64>& b1, const std::array<std::array<float, 64>, 64>& w2, const std::array<float, 64>& b2,
    const std::array<std::array<float, 64>, 4>& w3, const std::array<float, 4>& b3);

void updateParameters(const std::array<float, 4740>& flattened, std::array<std::array<float, 4>, 64>& w1, std::array<float, 64>& b1, std::array<std::array<float, 64>, 64>& w2, std::array<float, 64>& b2, 
    std::array<std::array<float, 64>, 4>& w3, std::array<float, 4>& b3);

void REINFORCEupdate(std::array<std::array<float, 4>, 64>& w1, std::array<float, 64>& b1, std::array<std::array<float, 64>, 64>& w2, std::array<float, 64>& b2, 
    std::array<std::array<float, 64>, 4>& w3, std::array<float, 4>& b3, const std::array<float, 4740>& gradientTerm);