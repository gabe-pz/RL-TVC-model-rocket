#pragma once 

#include "../math/rocketMath.h"


void slewServo(double& currentServoAngle, double desiredAngle, double maxRate, double dt);