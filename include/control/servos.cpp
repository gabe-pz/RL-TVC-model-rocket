#include "servos.h"


void slewServo(float& currentServoAngle, float desiredAngle, const double dt){
    constexpr float maxRate = 200.0f;//max angular v servo can move, in deg/sec

    float maxUpdate = maxRate*dt;//d(thetaMax) = omegaMax*dt, max angular movement servo can do in dt

    //essentially moving to desired angle within steps, and ensuring that each step is within the physical limits
    currentServoAngle += std::clamp(desiredAngle - currentServoAngle, -maxUpdate, maxUpdate);
}