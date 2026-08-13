#pragma once 

#include <fstream>
#include <array> 


void saveParameters(const std::array<std::array<float, 4>, 64>& w1, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 64>, 4>& w3, 
    const std::array<float, 64>& b1, const std::array<float, 64>& b2, const std::array<float, 4>& b3);

void loadParameters(const std::array<std::array<float, 4>, 64>& w1, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 64>, 4>& w3, 
    const std::array<float, 64>& b1, const std::array<float, 64>& b2, const std::array<float, 4>& b3);