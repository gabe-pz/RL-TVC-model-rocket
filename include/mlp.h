#pragma once

#include <random>
#include <array>

//*****MAIN FUNCTIONS*****
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


//*****HELPER FUNCTIONS FOR DERIVATIVES*****
float sumDerivatives(int k, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3, std::array<float, 64> y2);
std::array<float, 4> mKTerms(const std::array<float, 4>& outputs,  const std::array<float, 2>& actions);
std::array<float, 4> partialsSummed(const std::array<float, 64>& y2, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3);


float derivativeZkWrtW_1ij(const std::array<double, 4>& stateVec, const std::array<float, 64>& y1, const std::array<float, 4>& yOut, int i, int j, int k);
float derivativeZkWrtB_1ij(std::array<float, 64> y1, std::array<float, 4> yOut, int i, int k);

float derivativeZkWrtW_2ij(const std::array<float, 64>& y2, const std::array<float, 4>& yOut, const std::array<float, 64> a1, const std::array<std::array<float, 4>, 64>& w3, int i, int j, int k);
float derivativeZkWrtB_2ij(const std::array<float, 64>& y2, const std::array<float, 4>& yOut, const std::array<std::array<float, 4>, 64>& w3, int i, int k);


float derivativeZkWrtW_3ij(const std::array<float, 4>& yOut, const std::array<float, 64> a2, int j, int k);
float derivativeZkWrtB_3ij(const std::array<float, 4>& yOut, int k);


//*****GRADIENTS*****
//part 1 of gradients
std::array<std::array<float, 64*4>, 2> gradientLogPoliciesW1(const std::array<double, 4>& stateVec, std::array<float, 64>& y1, const std::array<float, 4>& yOut, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN);
//part 2 of gradients 
std::array<std::array<float, 64>, 2> gradientLogPoliciesB1(const std::array<float, 64>& y1, const std::array<float, 4>& outputs, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN);

//part 3 of gradients 
std::array<std::array<float, 64*64>, 2> gradientLogPoliciesW2(const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 64> a1, const std::array<float, 64> y2, const std::array<float, 4> yOut, const std::array<float, 4>& mK);
//part 4 of gradients 
std::array<std::array<float, 64>, 2> gradientLogPoliciesW2(const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 64> y2, const std::array<float, 4> yOut, const std::array<float, 4>& mK);

//part 5 of gradients 
std::array<std::array<float, 4*64>, 2> gradientLogPoliciesW3(const std::array<float, 4>& yOut, const std::array<float, 64> a2, const std::array<float, 4>& mK);
//part 6 of gradients
std::array<std::array<float, 4>, 2> gradientLogPoliciesB3(const std::array<float, 4>& yOut, const std::array<float, 4>& mK);

