#pragma once

#include <array> 
#include <cmath>

#include "windGeneration.h" 
#include "rocketProperties.h" 
#include "../math/rocketMath.h"


inline constexpr double centerOfGravity = 0.405;
inline constexpr double distanceToThrustVector = 0.6477;
void physicsUpdate(const double dt, double t, double U, double sigmaU, float actionX, float actionY, float servoXOffset, float servoYOffset, std::array<std::vector<double>, 3> pinkNoise, 
    std::array<double, 3>& position, std::array<double, 3>& velocity, std::array<double, 4>& stateQ, std::array<double, 3>& angularVelocity, std::array<double, 2>& psi);