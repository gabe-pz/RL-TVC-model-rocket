#pragma once

#include <array>

//Gradient term REINFORCE
std::array<float, 4740> gradientTerm(const std::array<float, 4740>& gradX, const std::array<float, 4740>& gradY, float alpha, float G);

//REINFORCE parameter update
void REINFORCEupdate(std::array<float, 4740>& parameters, const std::array<float, 4740>& gradientTerm);