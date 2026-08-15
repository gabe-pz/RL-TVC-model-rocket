#include <array>
#include <vector> 
#include <iostream> 
#include <random> 
#include <cmath> 
#include <algorithm> 

#include "../include/physics/windGeneration.h" 
#include "../include/math/rocketMath.h"
#include "../include/physics/rocketProperties.h"


#include "../include/rl/mlp.h"
#include "../include/rl/gradient.h"
#include "../include/rl/REINFORCE.h"
#include "../include/io/saveAndLoad.h"


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
    const double dt = 0.0001;
    const double simTime = tBurn;
    const double gravity = 9.81;  
    const float rho = 1.187f;
    double t = 0.0;

    //servos offsets 
    std::random_device rd;
    std::mt19937 gen(rd());
       

    //*****WIND SETTINGS*****
    //wind generation constants
    double U         = 5.0;  //average wind velocity
    double intensity = 0.20; //turbulence intensity 
    double sigmaU    = intensity * U;
    

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
    
    //mlp
    MLPoutput mlpOut;
    
    //network outputs, i.e distribution parameters
    std::array<float, 4> mlpControlOut; 
    
    //weights and biases initialization
    std::array<std::array<float, 4>, 64> w1; 
    std::array<std::array<float, 64>, 64> w2; 
    std::array<std::array<float, 64>, 4> w3; 
    std::array<float, 64> b1;
    std::array<float, 64> b2;
    std::array<float, 4> b3;
    initWeightsAndBiases(w1, w2, w3, b1, b2, b3);
        
    //actions
    float rawActionX = 0.0f;
    float rawActionY = 0.0f;
    float actionX = 0.0f;
    float actionY = 0.0f;

    //raw action output of the network 
    std::vector<std::array<float, 2>> rawActions;    
    
    //control 
    double controlDt = 0.05;
    double timeSinceLastControl = controlDt;

    //return and reward
    std::vector<double> reward;
    double returnT = 0.0; 

    //logging vars
    double returnAccumlated = 0.0;
    double accumlatedFlightTime = 0.0;
    double totalReturn = 0.0;

    //episodes and counters
    int numEpisodes = 25000;
    int numIterations = 0; 
    int episodesInWindow = 0;
    
    //hyperparameters
    float alpha = 0.0001f;//step size
    float gamma = 0.82f;//discount factor 
    float a = 125.0f;//exp constant
    // float b = 1.0f;//angular v penalize factor

    for(int e = 0; e < numEpisodes; e++){      
        //*****ENV*****

        //new random wind per episode
        unsigned int episodeSeed = rd();
        int n = (int)(simTime * GEN_FREQ) + 2;
        std::vector<double> pinkU = generatePinkNoise(n, episodeSeed);
        std::vector<double> pinkV = generatePinkNoise(n, episodeSeed + 1);
        std::vector<double> pinkW = generatePinkNoise(n, episodeSeed + 2);


        //*****RESET*****

        //time
        numIterations = 0;
        t = 0.0;
        timeSinceLastControl = controlDt;

        //state vars
        stateQ = {1.0, 0.0, 0.0, 0.0};
        stateQTimeDerivative = {0.0, 0.0, 0.0, 0.0};
        angularVelocity = {0.0, 0.0, 0.0};
        velocity = {0.0, 0.0, 0.0};
        position = {0.0, 0.0, 0.0};
        psi = {0.0, 0.0};

        //rl
        returnT = 0.0; 

        //storage
        rawActions.clear(); 
        reward.clear(); 
        stateVector.clear();

        //generate episode
        for(int i = 0; i < (int)(simTime/dt); i++){
            //time
            t += dt;
            timeSinceLastControl += dt;
            
            //*****CONTROL*****
            if(timeSinceLastControl >= controlDt){
                timeSinceLastControl = 0.0; 
                
                //ensure have ran inital action before getting inital reward
                if(numIterations > 0){

                    //episode termination check
                    if(std::abs(psi[0]) > 0.34 || std::abs(psi[1]) > 0.34){
                        
                        //negative reward for termination
                        reward.push_back(-10.0);       
                        break;
                    }
                
                    //reward update
                    double rewardT = std::exp(-a*(psi[0]*psi[0]+psi[1]*psi[1]));
                    reward.push_back(rewardT);       
                }

                //increment another step
                numIterations ++;

                //log and save state
                std::array<double, 4> state = {psi[0], psi[1], angularVelocity[0], angularVelocity[1]};
                stateVector.push_back(state);
                
                //mlp forward pass 
                mlpControlOut = mlpControl(state, w1, w2, w3, b1, b2, b3);

                //distribution parameters
                float muX = mlpControlOut[0];
                float sigmaX = std::exp(std::clamp(mlpControlOut[1], -1.0f, 2.0f));//ensure that the variance is positive by having network to output log of sigma and thus sigma is exp of that, which will keep it positive
                float muY = mlpControlOut[2];
                float sigmaY = std::exp(std::clamp(mlpControlOut[3], -1.0f, 2.0f));

                //distribution creation
                std::normal_distribution<float> dX{muX, sigmaX};
                std::normal_distribution<float> dY{muY, sigmaY}; 
                
                //action sampling
                rawActionX = dX(gen);
                rawActionY = dY(gen); 
                
                //clamp actions to domain of [-5 deg, 5 deg], where 0.087 rads is about 5 degs. 
                actionX = 0.087*std::tanh(rawActionX);
                actionY = 0.087*std::tanh(rawActionY);

                //log raw actions sampled from distribution
                rawActions.push_back({rawActionX, rawActionY});
            }


            //*****PHYSICS SIMULATION*****

            
            //*****WIND GEN*****
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
            
            //*****FORCES*****
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
            

            //****POSITION AND SUCH*****
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


            //*****TORQUES*****
            torqueThrust = crossProduct(r, thrustRf);
            torqueAero = crossProduct(rAero, aerodynamicForcesRf);


            //****ROTATION AND SUCH*****
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
        }
        
        //REINFORCE
        for(int i = 0; i < numIterations; i++){
        
            //return
            returnT = 0.0;
            for(int k = i; k < (int)reward.size(); k++){
                returnT += std::pow(gamma, k-i) * reward[k];
                
                //log total return from start of ep, for current ep
                if(i == 0 && k == (int)reward.size()-1){
                    totalReturn = returnT;
                }

            }


            //forward pass
            mlpOut = mlp(stateVector[i], w1, w2, w3, b1, b2, b3); 
            
            //pre-activations and activations, comptued for the new weights from the upated theta
            std::array<float, 64> y1 = mlpOut.y1;
            std::array<float, 64> y2 = mlpOut.y2;
            std::array<float, 4> y3 = mlpOut.y3;

            std::array<float, 64> a1 = mlpOut.a1;
            std::array<float, 64> a2 = mlpOut.a2;
            std::array<float, 4> a3 = mlpOut.a3; 
            
            //sigmas calculated for mkTerms
            float sigmaX = std::exp(std::clamp(a3[1], -1.0f, 2.0f));
            float sigmaY = std::exp(std::clamp(a3[3], -1.0f, 2.0f));

            //gradient calculation
            std::array<float, 4> mk = mKTerms(a3, {sigmaX, sigmaY}, rawActions[i]);
            std::array<std::array<float, 4>, 64> partialsN = partialsSummed(y2, w2, w3); 

            std::array<std::array<float, 64*4>, 2> gW1 = gradientLogPoliciesW1(stateVector[i], y1, y3, mk, partialsN);
            std::array<std::array<float, 64>, 2> gB1 = gradientLogPoliciesB1(y1, a3, mk, partialsN);
            std::array<std::array<float, 64*64>, 2> gW2 = gradientLogPoliciesW2(w3, a1, y2, y3, mk);
            std::array<std::array<float, 64>, 2> gB2 = gradientLogPoliciesB2(w3, y2, y3, mk);
            std::array<std::array<float, 4*64>, 2> gW3 = gradientLogPoliciesW3(y3, a2, mk);
            std::array<std::array<float, 4>, 2> gB3 = gradientLogPoliciesB3(y3, mk);

            //construct the gradient
            std::array<std::array<float, 4740>, 2> constructGrads = constructGradients(gW1, gB1, gW2, gB2, gW3, gB3);
            std::array<float, 4740> gradX = constructGrads[0];
            std::array<float, 4740> gradY = constructGrads[1];
            
            //calculate the gradient term in REINFORCE
            std::array<float, 4740> gradTerm = gradientTerm(gradX, gradY, alpha, returnT);
            
            //update parameters with gradient term
            std::array<float, 4740> parameters = flattenParameters(w1, b1, w2, b2, w3, b3);

            REINFORCEupdate(parameters, gradTerm);
                    
            updateParameters(parameters, w1, b1, w2, b2, w3, b3);
        }

        //for logging purposes
        returnAccumlated += totalReturn;
        accumlatedFlightTime += numIterations*controlDt;
        episodesInWindow ++; 

        //Log to terminal every 1k eps some data 
        if((e % 1000 == 0 && e > 0) || e == numEpisodes - 1){
            std::cout << "*************************************************************" << std::endl;
            std::cout << "DATA AFTER " << e << " EPISODES: " << std::endl; 
            std::cout << "AVERAGE RETURN = " << (returnAccumlated / episodesInWindow) << std::endl;
            std::cout << "AVERAGE FLIGHT TIME = " << (accumlatedFlightTime / episodesInWindow) << "s" << std::endl;

            returnAccumlated = 0.0;
            accumlatedFlightTime = 0.0;
            episodesInWindow = 0.0;
        }
    }

    //save learned parameters to .bin
    saveParameters(w1, w2, w3, b1, b2, b3);
    
    std::cout << w1[1][1] << std::endl;
    return 0;
}
