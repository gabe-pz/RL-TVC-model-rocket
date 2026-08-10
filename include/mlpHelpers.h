#pragma once

#include <random>
#include <array>

void initWeightsAndBiases(std::array<float, 64*4>& w1, std::array<float, 64*64>& w2, std::array<float, 4*64>& w3, std::array<float, 64>& b1, std::array<float, 64>& b2, std::array<float, 4>& b3); 

float preActivation();
float ReLU(float y);