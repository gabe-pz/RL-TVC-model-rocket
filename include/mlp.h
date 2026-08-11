#pragma once

#include <random>
#include <array>

//*****ACTIVATION FUNCTION*****
float ReLU(float y);


//*****FUNCTIONS FOR PARAMETERS*****
void initWeightsAndBiases(std::array<std::array<float, 4>, 64>& w1, std::array<std::array<float, 64>, 64>& w2, std::array<std::array<float, 4>, 64>& w3, std::array<float, 64>& b1, std::array<float, 64>& b2, std::array<float, 4>& b3); 
std::array<float, 4740> flatten_parameters(const std::array<std::array<float, 4>, 64>& w1, const std::array<float, 64>& b1, const std::array<std::array<float, 64>, 64>& w2, const std::array<float, 64>& b2,
    const std::array<std::array<float, 4>, 64>& w3,
    const std::array<float, 4>& b3);

//*****Pre-activations and activations*****
template<typename T, std::size_t in, std::size_t out>
std::array<float, out> preActivations(const std::array<std::array<float, in>, out>& w, const std::array<float, out>& b, const std::array<T, in>& vecIn);

template<std::size_t out>
std::array<float, out> activations(std::array<float, out> y1);
