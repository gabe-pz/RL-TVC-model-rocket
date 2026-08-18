#include "servos.h"


void slewServo(float& currentServoAngle, float desiredAngle, const double dt){
    constexpr float maxRate = 3.491f;//max angular v servo can move, in rads/sec

    float maxUpdate = maxRate*dt;//d(thetaMax) = omegaMax*dt, max angular movement servo can do in dt

    //essentially moving to desired angle within steps if need be, that is if d(theta) is greater than d(thetaMax)
    currentServoAngle += std::clamp(desiredAngle - currentServoAngle, -maxUpdate, maxUpdate);
}