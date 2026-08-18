#pragma once 

#include <algorithm>

#include "../math/rocketMath.h"


void slewServo(float& currentServoAngle, float desiredAngle, const double dt);