#include "servos.h"


void slewServo(double& currentServoAngle, double desiredAngle, double maxRate, double dt){
    //d(theta) = omega*dt, max angular movement servo can do in dt
    double maxUpdate = maxRate*dt;

    //essentially moving to desired angle within steps, and ensuring that each step is within the physical limits
    currentServoAngle += clamp(desiredAngle - currentServoAngle, -maxUpdate, maxUpdate);
}