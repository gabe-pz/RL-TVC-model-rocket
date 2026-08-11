#pragma once 

#include <array>

//*****GENERAL DERIVATIVES*****
float derivativeActivationWrtPreactivation(float yLk);

template<std::size_t in, std::size_t out>
float derivativePreactivationWrtActivation(const std::array<std::array<float, in>, out>& w, int n, int m);

template<typename T, std::size_t in>
float derivativePreactivationWrtWeight(const std::array<T, in>& vecIn, int j);


//*****HELPER FUNCTIONS FOR DERIVATIVES*****
float sumDerivatives(int k, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3, std::array<float, 64> y2);
std::array<float, 4> mKTerms(const std::array<float, 4>& outputs,  const std::array<float, 2>& actions);
std::array<float, 4> partialsSummed(const std::array<float, 64>& y2, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3);

//*****PARTICULAR DERIVATIVES*****
float derivativeZkWrtW_1ij(const std::array<double, 4>& stateVec, const std::array<float, 64>& y1, const std::array<float, 4>& yOut, int i, int j, int k);
float derivativeZkWrtB_1ij(std::array<float, 64> y1, std::array<float, 4> yOut, int i, int k);

float derivativeZkWrtW_2ij(const std::array<float, 64>& y2, const std::array<float, 4>& yOut, const std::array<float, 64> a1, const std::array<std::array<float, 4>, 64>& w3, int i, int j, int k);
float derivativeZkWrtB_2ij(const std::array<float, 64>& y2, const std::array<float, 4>& yOut, const std::array<std::array<float, 4>, 64>& w3, int i, int k);


float derivativeZkWrtW_3ij(const std::array<float, 4>& yOut, const std::array<float, 64> a2, int j, int k);
float derivativeZkWrtB_3ij(const std::array<float, 4>& yOut, int k);


//*****GRADIENT*****
//part 1 of gradients
std::array<std::array<float, 64*4>, 2> gradientLogPoliciesW1(const std::array<double, 4>& stateVec, std::array<float, 64>& y1, const std::array<float, 4>& yOut, const std::array<float, 4>& mK, 
    const std::array<float, 4>& partialsN);
//part 2 of gradients 
std::array<std::array<float, 64>, 2> gradientLogPoliciesB1(const std::array<float, 64>& y1, const std::array<float, 4>& outputs, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN);

//part 3 of gradients 
std::array<std::array<float, 64*64>, 2> gradientLogPoliciesW2(const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 64> a1, const std::array<float, 64> y2, const std::array<float, 4> yOut, 
    const std::array<float, 4>& mK);
//part 4 of gradients 
std::array<std::array<float, 64>, 2> gradientLogPoliciesB2(const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 64> y2, const std::array<float, 4> yOut, const std::array<float, 4>& mK);

//part 5 of gradients 
std::array<std::array<float, 4*64>, 2> gradientLogPoliciesW3(const std::array<float, 4>& yOut, const std::array<float, 64> a2, const std::array<float, 4>& mK);
//part 6 of gradients
std::array<std::array<float, 4>, 2> gradientLogPoliciesB3(const std::array<float, 4>& yOut, const std::array<float, 4>& mK);

//construction of gradient
std::array<std::array<float, 4740>, 2> constructGradients(const std::array<std::array<float, 64*4>, 2>& gW1, const std::array<std::array<float, 64>, 2>& gB1, const std::array<std::array<float, 64*64>, 2>& gW2, 
    const std::array<std::array<float, 64>, 2>& gB2, const std::array<std::array<float, 4*64>, 2>& gW3, const std::array<std::array<float, 4>, 2>& gB3);
