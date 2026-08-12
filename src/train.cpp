#include <array>
#include <vector> 
#include <iostream> 
#include <random> 

#include "../include/physics/windGeneration.h" 
#include "../include/math/rocketMath.h"
#include "../include/physics/rocketProperties.h"
#include "../include/control/control.h"

#include "../include/rl/mlp.h"
#include "../include/rl/gradient.h"
#include "../include/rl/REINFORCE.h"
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
    const int simTime = tBurn;
    const double gravity = 9.81;  
    const float rho = 1.187f;
    double t = 0.0;

    //*****SERVO SETTINGS****
    double currentServoX = 0.0;
    double currentServoY = 0.0;

    //servos offsets 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 2.25f);//misalignment between 0.5 and 1 degree
    float servosXOffset = dist(gen);
    float servosYOffset = dist(gen);
       

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


    //*****RL*****
    //state vector
    std::vector<std::array<double, 4>> stateVector;
    
    //mlp outputs
    MLPoutput mlpOut;

    std::vector<std::array<float, 64>> y1, a1, y2, a2; 
    std::vector<std::array<float, 4>> y3, a3;  

    
    //weights and biases initialization
    std::array<std::array<float, 4>, 64> w1; 
    std::array<std::array<float, 64>, 64> w2; 
    std::array<std::array<float, 64>, 4> w3; 
    std::array<float, 64> b1;
    std::array<float, 64> b2;
    std::array<float, 4> b3;
    initWeightsAndBiases(w1, w2, w3, b1, b2, b3);
    
    //actions
    float actionX;
    float actionY;
    std::vector<std::array<float, 2>> actions;

    //return and episodes
    int totalReturn = 0;
    std::vector<int> reward;
    
    int numEpisodes = 1000;
    int numIterations = 0; 
    float alpha = 0.02;
    
    //probability distribution    
    float muX;
    float sigmaX;
    float muY;
    float sigmaY;


    for(int e = 0; e < numEpisodes; e++){        
        //RESET
        numIterations = 0;
        t = 0.0;
        stateQ = {1.0, 0.0, 0.0, 0.0};
        stateQTimeDerivative = {0.0, 0.0, 0.0, 0.0};
        angularVelocity = {0.0, 0.0, 0.0};
        velocity = {0.0, 0.0, 0.0};
        position = {0.0, 0.0, 0.0};
        psi = {0.0, 0.0};
        totalReturn = 0;
        actions.clear(); 
        reward.clear(); 

        //generate episode
        for(int i = 0; i < simTime; i++){
            //time and iterations
            t += dt;
            numIterations ++;
            
            //mlp forward pass
            std::array<double, 4> state = {psi[0], psi[1], angularVelocity[0], angularVelocity[1]};
            stateVector.push_back(state);
            mlpOut = mlp(state, w1, w2, w3, b1, b2, b3);

            //pre-activations and activations(to-do: cache these such that can do REINFORCE at end of ep)
            y1.push_back(mlpOut.y1);
            a1.push_back(mlpOut.a1);
            y2.push_back(mlpOut.y2);
            a2.push_back(mlpOut.a2);
            y3.push_back(mlpOut.y3);
            a3.push_back(mlpOut.a3);

            //distribution parameters
            muX = mlpOut.a3[0];
            sigmaX = std::exp(mlpOut.a3[1]); 
            muY = mlpOut.a3[2];
            sigmaY = std::exp(mlpOut.a3[3]);; 
            
            //action sampling 
            std::normal_distribution<float> dX{muX, sigmaX};
            std::normal_distribution<float> dY{muY, sigmaY}; 

            actionX = dX(gen);
            actionY = dY(gen);

            actions.push_back({actionX, actionY});

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
            thrustRf = forceThrustRf(deg2rad(actionX+servosXOffset), deg2rad(actionY+servosYOffset), t);
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

            //conver to euler angles
            psi = quaternionToEuler(stateQ); 

            //reward update
            if(psi[0] < 0.2 && psi[1] < 0.2){
                reward.push_back(1);
            }
            else break;
        }
        //total return for episode ran
        for(int i = 0; i < (int)reward.size(); i++){ 
            totalReturn += reward[i];
        }

        for(int i = 0; i < numIterations; i++){
            std::array<float, 4> mk = mKTerms(a3[i], actions[i]);
            std::array<std::array<float, 4>, 64> partialsN = partialsSummed(y2[i], w2, w3); 

            std::array<std::array<float, 64*4>, 2> gW1 = gradientLogPoliciesW1(stateVector[i], y1[i], y3[i], mk, partialsN);
            std::array<std::array<float, 64>, 2> gB1 = gradientLogPoliciesB1(y1[i], a3[i], mk, partialsN);
            std::array<std::array<float, 64*64>, 2> gW2 = gradientLogPoliciesW2(w3, a1[i], y2[i], y3[i], mk);
            std::array<std::array<float, 64>, 2> gB2 = gradientLogPoliciesB2(w3, y2[i], y3[i], mk);

            std::array<std::array<float, 4*64>, 2> gW3 = gradientLogPoliciesW3(y3[i], a2[i], mk);
            std::array<std::array<float, 4>, 2> gB3 = gradientLogPoliciesB3(y3[i], mk);

            std::array<std::array<float, 4740>, 2> constructGrads = constructGradients(gW1, gB1, gW2, gB2, gW3, gB3);

            std::array<float, 4740> gradX = constructGrads[0];
            std::array<float, 4740> gradY = constructGrads[1];

            std::array<float, 4740> gradTerm = gradientTerm(gradX, gradY, alpha, totalReturn);

            std::array<float, 4740> parameters = flattenParameters(w1, b1, w2, b2, w3, b3);

            REINFORCEupdate(parameters, gradTerm);

            unflattenParameters(parameters, w1, b1, w2, b2, w3, b3);
        }

    }
    


    return 0;
}