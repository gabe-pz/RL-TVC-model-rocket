#pragma once

#include <raylib.h> 
#include <rlgl.h>  
#include "../math/rocketMath.h"

Camera3D raylibInit(int FPS);
void raylibDrawRocket(const double distanceToThrustVector, const double centerOfGravity, const std::array<double, 4>& stateQ, const std::array<double, 3> position, Camera3D& camera);