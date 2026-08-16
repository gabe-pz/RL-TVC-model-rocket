#include <iostream>
#include <random> 
#include <array>
#include <fstream>

#include "../include/physics/physicsLoop.h"
#include "../include/control/rlControl.h"
#include "../include/io/log.h"
#include "../include/io/saveAndLoad.h"
#include "../include/graphics/raylibFunction.h"

int main(void){     
    //*****RANDOM NUM GEN*****
    std::random_device rd;
    std::mt19937 gen(rd());
    
    //*****SIMULATION SETTINGS*****
    const double dt = 0.0001;
    const int simTime = 15;
    double t = 0.0;

    //*****WIND SETTINGS*****
    //wind generation constants
    unsigned int seed = 12345;
    int n = (int)(simTime * GEN_FREQ) + 2;
    double U = 5.0;  //average wind velocity
    double intensity = 0.20; //turbulence intensity 
    double sigmaU = intensity * U;
    
    //wind turbulence buffers
    std::vector<double> pinkU = generatePinkNoise(n, seed);
    std::vector<double> pinkV = generatePinkNoise(n, seed + 1);
    std::vector<double> pinkW = generatePinkNoise(n, seed + 2);
    std::array<std::vector<double>, 3> pinkNoise = {pinkU, pinkV, pinkW};

    //*****LOGGING*****
    double logInterval = 0.01; 
    double timeSinceLastLog = logInterval; 
    initCSV("logging/dataRotX.csv", 0);
    initCSV("logging/dataRotY.csv", 1);

    //*****STATE CHECKS VARS*****
    double currentAltitude = 0.0;
    double prevAltitude = 0.0;
    bool landed = false;
    bool printedApogee = false; 


    //*****RAYLIB INITALIZATION*****
    int FPS = 60;  
    Camera3D camera = raylibInit(FPS);
    //raylib sync 
    int iterationsPerFrame = (1/(dt*FPS)); 


    //*******STATE ARRAY INITALIZATIONS*****
    //quaterion initaliztion
    std::array<double, 4> stateQ = {1.0, 0.0, 0.0, 0.0};
    
    std::array<double, 3> velocity = {0.0, 0.0, 0.0};
    std::array<double, 3> position = {0.0, 0.0, 0.0};

    std::array<double, 3> angularVelocity = {0.0, 0.0, 0.0};
    std::array<double, 2> psi ={0.0, 0.0};


    //*****SERVOS*****
    
    //servos offset
    std::uniform_real_distribution<float> dist(0.017f, 0.034f);//misalignment between 1 and 2 deg
    float servosXOffset = dist(gen);
    float servosYOffset = dist(gen);

    //*****CONTROL*****

    float actionX = 0.0f;
    float actionY = 0.0f;

    //speed to run control at
    const double controlDt = 0.05; 
    double timeSinceLastControl = controlDt; 

    //weights and biases for network
    std::array<std::array<float, 4>, 64> w1; 
    std::array<std::array<float, 64>, 64> w2; 
    std::array<std::array<float, 64>, 4> w3; 
    std::array<float, 64> b1;
    std::array<float, 64> b2;
    std::array<float, 4> b3;
    loadParameters(w1, w2, w3, b1, b2, b3);

    while (!WindowShouldClose()){
        if(!landed){
            //physics loop
            for(int i = 0; i < iterationsPerFrame; i++){
                //*****TIME*****
                t += dt;
                timeSinceLastControl += dt; 

                //******CONTROL*****
                if(timeSinceLastControl >= controlDt){
                    std::array<double, 4> stateVector = {psi[0], psi[1], angularVelocity[0], angularVelocity[1]};

                    std::array<float, 2> policyOutputs = policy(stateVector, w1, w2, w3, b1, b2, b3);

                    float rawActionX = policyOutputs[0];
                    float rawActionY = policyOutputs[1];

                    actionX = 0.087*std::tanh(rawActionX);
                    actionY  = 0.087*std::tanh(rawActionY);

                    timeSinceLastControl = 0.0;
                }
                
                //******PHYSICS UPDATE******
                physicsUpdate(dt, t, U, sigmaU, actionX, actionY, servosXOffset, servosYOffset, pinkNoise, position, velocity, stateQ, angularVelocity, psi);
                                
                //*****LANDING CHECK*****
                if(position[2] < 0 && t > 0.5){
                    landed = true; 
                    break;
                }
                
                //*****APOGEE*****
                currentAltitude = position[2];
                if(currentAltitude < prevAltitude && printedApogee != true){
                    std::cout << "APOGEE at t = " << t << "s" << std::endl; 
                    std::cout << "APOGEE ALTITUDE = " << position[2]*3.281 << "ft" << std::endl;
                    printedApogee = true;
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

    return 0;
}
