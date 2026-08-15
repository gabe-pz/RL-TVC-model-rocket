#include <iostream>
#include <random> 
#include <array>
#include <fstream>

#include "../include/physics/windGeneration.h" 
#include "../include/math/rocketMath.h"
#include "../include/physics/rocketProperties.h"
#include "../include/control/rlControl.h"
#include "../include/io/log.h"
#include "../include/io/saveAndLoad.h"
#include "../include/graphics/raylibFunction.h"

int main(void){     
    //*****ROCKET PROPERTIES*****
    //aerodynamic constants
    const double centerOfPressure = 0.0877;
    const double aRef = 0.00456; 
    const double cD = 0.291;
    const int cNa = 2;

    //cg and moment arm
    const double centerOfGravity = 0.405;
    const double distanceToThrustVector = 0.6477;

    //moment of inertia
    long double Ixx = 0.0249899588;
    long double Iyy = 0.0249868814;
    
    //*****SIMULATION SETTINGS*****
    const double dt = 0.000001;
    const int simTime = 15;
    const double gravity = 9.81;  
    const float rho = 1.187f;

    //*****WIND SETTINGS*****
    //wind generation constants
    unsigned int seed = 12345;
    int n = (int)(simTime * GEN_FREQ) + 2;
    double U         = 5.0;  //average wind velocity
    double intensity = 0.20; //turbulence intensity 
    double sigmaU    = intensity * U;
    
    //wind turbulence buffers
    std::vector<double> pinkU = generatePinkNoise(n, seed);
    std::vector<double> pinkV = generatePinkNoise(n, seed + 1);
    std::vector<double> pinkW = generatePinkNoise(n, seed + 2);

    //mean-wind heading in world frame
    double theta = 0.0;
    double ux = std::cos(theta);
    double uy = std::sin(theta);

    //*****LOGGING*****
    double logInterval = 0.01; 
    double timeSinceLastLog = logInterval; 
    initCSV("logging/dataRotX.csv", 0);
    initCSV("logging/dataRotY.csv", 1);

    //*****STATE CHECKS VARS*****
    double currentAltitude = 0.0;
    double prevAltitude = 0.0;
    bool landed = false;
    bool coastingOver = false;
    bool printedCoasted = false;
    bool printedApogee = false; 

    //*****CONTROL*****
    std::array<float, 2> policyOutputs;
    
    float rawActionX = 0.0f;
    float rawActionY = 0.0f;
    
    float actionX = 0.0f;
    float actionY = 0.0f;
    

    //speed to run control at
    const double controlDt = 0.05; 
    double timeSinceLastControl = controlDt; 

    //weights and biases
    std::array<std::array<float, 4>, 64> w1; 
    std::array<std::array<float, 64>, 64> w2; 
    std::array<std::array<float, 64>, 4> w3; 
    std::array<float, 64> b1;
    std::array<float, 64> b2;
    std::array<float, 4> b3;
    loadParameters(w1, w2, w3, b1, b2, b3);
    std::cout << w1[1][1] << std::endl;
    

    //*****RAYLIB INITALIZATION*****
    int FPS = 60;  
    Camera3D camera = raylibInit(FPS);
    //raylib sync 
    int iterationsPerFrame = (1/(dt*FPS)); 
    double t = 0.0;


    //*******STD ARRAY INITALIZATIONS*****
    //quaterion initaliztion
    std::array<double, 4> stateQ = {1.0, 0.0, 0.0, 0.0};
    std::array<double, 4> stateQTimeDerivative = {0.0, 0.0, 0.0, 0.0};
    std::array<double, 4> angularVelocityQ = {1.0, 0.0, 0.0, 0.0};

    //forces initalization
    std::array<double, 3> thrustRf = {0.0, 0.0, 0.0};
    std::array<double, 3> thrustWf = {0.0, 0.0, 0.0};
    std::array<double, 3> aerodynamicForcesRf = {0.0, 0.0, 0.0}; 
    std::array<double, 3> aerodynamicForceswf = {0.0, 0.0, 0.0}; 
    std::array<double, 3> sumOfForcesWf = {0.0, 0.0, 0.0}; 

    //torques initalization
    std::array<double, 3> torqueThrust = {0.0, 0.0, 0.0}; 
    std::array<double, 3> torqueAero = {0.0, 0.0, 0.0}; 
    
    //position and its derivatives initalization
    std::array<double, 3> accleration = {0.0, 0.0, 0.0};
    std::array<double, 3> velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> position = {0.0, 0.0, 0.0}; 
    std::array<double, 3> relativeVelocityWf = {0.0, 0.0, 0.0};
    std::array<double, 3> relativeVelocityRf = {0.0, 0.0, 0.0};

    //rotation and its derivatives initalization. Note psi = (phi, theta)
    std::array<double, 3> angularAccleration = {0.0, 0.0, 0.0};
    std::array<double, 3> angularVelocity = {0.0, 0.0, 0.0};
    std::array<double, 2> psi = {0.0, 0.0};
    
    //moment arms
    std::array<double, 3> r = {0.0, 0.0, centerOfGravity-distanceToThrustVector};
    std::array<double, 3> rAero = {0.0, 0.0, centerOfGravity-centerOfPressure};

    //wind velocity initalization
    std::array<double, 3> windVelocityWf = {0.0, 0.0, 0.0};


    std::array<double, 4> stateVector;

    while (!WindowShouldClose()){
        if(!landed){
            //physics loop
            for(int i = 0; i < iterationsPerFrame; i++){
                //*****TIME*****
                t += dt;
                timeSinceLastControl += dt; 

                //******PID CONTROL*****z
                if(timeSinceLastControl >= controlDt){
                    stateVector = {psi[0], psi[1], angularVelocity[0], angularVelocity[1]};

                    policyOutputs = policy(stateVector, w1, w2, w3, b1, b2, b3);

                    rawActionX = policyOutputs[0];
                    rawActionY = policyOutputs[1];

                    actionX = 0.087*std::tanh(rawActionX);
                    actionY = 0.087*std::tanh(rawActionY);

                    timeSinceLastControl = 0.0;
                }

                //*****WIND*****
                //sampling three independent turbulence streams at time t
                double u = windVelocity(t, U,   sigmaU,       pinkU);   
                double v = windVelocity(t, 0.0, 0.8 * sigmaU, pinkV);   
                double w = windVelocity(t, 0.0, 0.5 * sigmaU, pinkW);  

                //rotate wind frame -> world frame
                windVelocityWf = {u*ux - v*uy, u*uy + v*ux, w};
                
                //velocity of rocket wrt to wind in world frame, then into rocket frame
                relativeVelocityWf[0] = velocity[0] - windVelocityWf[0];
                relativeVelocityWf[1] = velocity[1] - windVelocityWf[1];
                relativeVelocityWf[2] = velocity[2] - windVelocityWf[2];
                relativeVelocityRf = rotateWfToRf(stateQ, relativeVelocityWf);
                
                //magnitude of relative v in RF
                double relativeVelMag = std::sqrt((relativeVelocityRf[0]*relativeVelocityRf[0]) + (relativeVelocityRf[1]*relativeVelocityRf[1]) + (relativeVelocityRf[2]*relativeVelocityRf[2]));
                
                //*****COMPUTE FORCES*****
                //force due to thrust
                thrustRf = forceThrustRf(actionX, actionY, t);
                thrustWf = rotateRfToWf(stateQ, thrustRf); 

                //aero forces. Note for normal force throw the negative on there to account for the fact want to use the free-stream velocity
                aerodynamicForcesRf[0] = -0.5*rho*cNa*aRef*relativeVelocityRf[0]*relativeVelMag;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                aerodynamicForcesRf[1] = -0.5*rho*cNa*aRef*relativeVelocityRf[1]*relativeVelMag;
                aerodynamicForcesRf[2] = -0.5*rho*cD*aRef*(std::abs(relativeVelocityRf[2]))*relativeVelocityRf[2]; 
                aerodynamicForceswf = rotateRfToWf(stateQ, aerodynamicForcesRf);
            
                //sum forces
                sumOfForcesWf = {thrustWf[0]+aerodynamicForceswf[0], thrustWf[1]+aerodynamicForceswf[1], thrustWf[2]-mass(t)*gravity+aerodynamicForceswf[2]}; 
                

                //****GET POSITION THROUGH ITS DERIVATIVES VIA EULER INTEGRATION*****
                //compute accleration
                accleration[0] = (sumOfForcesWf[0] / mass(t));
                accleration[1] = (sumOfForcesWf[1] / mass(t));
                accleration[2] = (sumOfForcesWf[2] / mass(t));

                //integrate accleration for velocity
                velocity[0] += dt*accleration[0];
                velocity[1] += dt*accleration[1];
                velocity[2] += dt*accleration[2];

                //integrate velocity for position 
                position[0] += dt*velocity[0];
                position[1] += dt*velocity[1];
                position[2] += dt*velocity[2]; 


                //*****COMPUTE TORQUES*****
                torqueThrust = crossProduct(r, thrustRf);
                torqueAero = crossProduct(rAero, aerodynamicForcesRf);


                //****GET ROATION THROUGH ITS DERIVATIVES VIA EULER INTEGRATION*****
                //compute angular accleration
                angularAccleration[0] = ((torqueThrust[0] + torqueAero[0]) / Ixx);
                angularAccleration[1] = ((torqueThrust[1] + torqueAero[1]) / Iyy);

                //integrate angular accleration for angular velocity 
                angularVelocity[0] += dt*angularAccleration[0];
                angularVelocity[1] += dt*angularAccleration[1];

                //convert angular velocity to pure quaternion
                angularVelocityQ = vectorToPureQuaternion(angularVelocity);

                //compute first derivative of quaternion
                stateQTimeDerivative = multiplyQP(stateQ, angularVelocityQ);
                stateQTimeDerivative[0] = stateQTimeDerivative[0]*0.5;
                stateQTimeDerivative[1] = stateQTimeDerivative[1]*0.5;
                stateQTimeDerivative[2] = stateQTimeDerivative[2]*0.5;
                stateQTimeDerivative[3] = stateQTimeDerivative[3]*0.5;

                //integrate time derivative of q to update state quaternion
                stateQ[0] += dt*stateQTimeDerivative[0];
                stateQ[1] += dt*stateQTimeDerivative[1];
                stateQ[2] += dt*stateQTimeDerivative[2];
                stateQ[3] += dt*stateQTimeDerivative[3];

                //normalize quaterinon 
                normalizeQuaternion(stateQ);

                //EULER ANGLES
                psi = quaternionToEuler(stateQ); 
                
                
                //*****STATE CHECKS*****
                
                    //*****LANDING CHECK*****
                    if(position[2] < 0 && t > 0.5){
                        landed = true; 
                        break;
                    }
                    
                    //******COASTING*****
                    if(t > 3.45 && coastingOver != true && printedCoasted != true){
                        std::cout << "COASTING PHASE" << std::endl;
                        printedCoasted = true;
                    }

                    //*****APOGEE*****
                    currentAltitude = position[2];
                    if(currentAltitude < prevAltitude && printedApogee != true){
                        std::cout << "APOGEE at t = " << t << "s" << std::endl; 
                        std::cout << "APOGEE ALTITUDE = " << position[2]*3.281 << "ft" << std::endl;
                        printedApogee = true;
                        coastingOver = true;
                    }
                    prevAltitude = currentAltitude;

                //*****LOGGING*****
                timeSinceLastLog += dt;
                if(timeSinceLastLog >= logInterval){
                    logToCSV(t, rad2deg(psi[0]), "logging/dataRotX.csv");
                    logToCSV(t, rad2deg(psi[1]), "logging/dataRotY.csv");
                    timeSinceLastLog = 0.0; 
                }
                
            }
        }
        raylibDrawRocket(distanceToThrustVector, centerOfGravity, stateQ, position, camera);
    }

    CloseWindow();    

    //final print
    std::cout << "Final position: (" << position[0] << "x, " << position[1] << "y, " << position[2] << "z)" << std::endl;

    return 0;
}
