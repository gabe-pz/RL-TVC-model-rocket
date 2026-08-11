#pragma once

#include <random>
#include <array>

float ReLU(float y);

void initWeightsAndBiases(std::array<std::array<float, 4>, 64>& w1, std::array<std::array<float, 64>, 64>& w2, std::array<std::array<float, 4>, 64>& w3, std::array<float, 64>& b1, std::array<float, 64>& b2, std::array<float, 4>& b3); 

//*****Pre-activations and activations*****
template<typename T, std::size_t in, std::size_t out>
std::array<float, out> preActivations(const std::array<std::array<float, in>, out>& w, const std::array<float, out>& b, const std::array<T, in>& vecIn);

template<std::size_t out>
std::array<float, out> activations(std::array<float, out> y1);

//*****DERIVATIVES*****
float derivativeActivationWrtPreactivation(float yLk);

template<std::size_t in, std::size_t out>
float derivativePreactivationWrtActivation(const std::array<std::array<float, in>, out>& w, int n, int m);

template<typename T, std::size_t in>
float derivativePreactivationWrtWeight(const std::array<T, in>& vecIn, int j);

//sum term
float sumDerivatives(int k, const std::array<std::array<float, 4>, 64>& w1, const std::array<std::array<float, 64>, 64>& w2, std::array<float, 64> y2);

//Derivatives of Zk Wrt params
float derivativeZkWrtW_1ij(const std::array<double, 4>& stateVec, std::array<float, 64> y1, std::array<float, 4> outputs, int i, int j, int k);
float derivativeZkWrtB_1ij(std::array<float, 64> y1, std::array<float, 4> outputs, int i, int k);

std::array<std::array<float, 4>, 2> mKAndPartialSummed(const std::array<float, 4>& outputs, const std::array<float, 64>& y2, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 2>& actions);

//*****GRADIENTS*****

//part 1
std::array<std::array<float, 64*4>, 2> gradientLogPoliciesW1(const std::array<double, 4>& stateVec, std::array<float, 64>& y1, const std::array<float, 4>& outputs, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN);

//part 2
std::array<std::array<float, 64>, 2> gradientLogPoliciesB1(const std::array<float, 64>& y1, const std::array<float, 4>& outputs, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN);