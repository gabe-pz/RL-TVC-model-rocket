#include <array>
#include <vector> 
#include <iostream> 
#include <random> 
#include <cmath> 

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

    std::vector<std::array<float, 64>> y1, a1, y2, a2; 
    std::vector<std::array<float, 4>> y3, a3;  
    
    //network outputs
    std::vector<std::array<float, 4>> outputs; 
    
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

    //storing the raw outputs for mkTerms
    std::vector<std::array<float, 2>> rawActions;

    //return and reward
    double totalReturn = 0.0;
    std::vector<double> reward;
    double accumlatedReturn = 0.0;
   

    //episodes and counter
    int numEpisodes = 1000000;
    int numIterations = 0; 
    
    //hyperparameters
    float alpha = 0.0001f;
    float gamma = 0.75f;
    float a = 3.0f;

    //control 
    double controlDt = 0.05;
    double timeSinceLastControl = controlDt;

    for(int e = 0; e < numEpisodes; e++){      
        //new random wind per episode
        unsigned int episodeSeed = rd();
        int n = (int)(simTime * GEN_FREQ) + 2;
        std::vector<double> pinkU = generatePinkNoise(n, episodeSeed);
        std::vector<double> pinkV = generatePinkNoise(n, episodeSeed + 1);
        std::vector<double> pinkW = generatePinkNoise(n, episodeSeed + 2);


        //RESET
        numIterations = 0;
        t = 0.0;
        timeSinceLastControl = controlDt;

        stateQ = {1.0, 0.0, 0.0, 0.0};
        stateQTimeDerivative = {0.0, 0.0, 0.0, 0.0};
        angularVelocity = {0.0, 0.0, 0.0};
        velocity = {0.0, 0.0, 0.0};
        position = {0.0, 0.0, 0.0};
        psi = {0.0, 0.0};

        totalReturn = 0.0; 

        rawActions.clear(); 
        reward.clear(); 
        y1.clear(); a1.clear();
        y2.clear(); a2.clear();
        y3.clear(); a3.clear();
        stateVector.clear();
        outputs.clear();

        //generate episode
        for(int i = 0; i < (int)(simTime/dt); i++){
            //time and iterations
            t += dt;
            timeSinceLastControl += dt;

            if(timeSinceLastControl >= controlDt){
                timeSinceLastControl = 0.0; 
                
                if(numIterations > 0){

                    if(std::abs(psi[0]) > 0.2 || std::abs(psi[1]) > 0.2){
                        reward.push_back(-10);         
                        break;
                    }
                    
                    //reward update
                    reward.push_back(std::exp(-a*(psi[0]*psi[0])) + std::exp(-a*(psi[1]*psi[1])));         
                }

                //increment another step
                numIterations ++;

                //mlp forward pass
                std::array<double, 4> state = {psi[0], psi[1], angularVelocity[0], angularVelocity[1]};
                stateVector.push_back(state);
                mlpOut = mlp(state, w1, w2, w3, b1, b2, b3);

                //pre-activations and activations
                y1.push_back(mlpOut.y1);
                a1.push_back(mlpOut.a1);
                y2.push_back(mlpOut.y2);
                a2.push_back(mlpOut.a2);
                y3.push_back(mlpOut.y3);
                a3.push_back(mlpOut.a3);


                //distribution parameters
                float muX = mlpOut.a3[0];
                float sigmaX = std::exp(mlpOut.a3[1]);//ensure that the variance is positive by having network to output log of sigma and thus sigma is exp of that, which will keep it positive
                float muY = mlpOut.a3[2];
                float sigmaY = std::exp(mlpOut.a3[3]);
                outputs.push_back({muX, sigmaX, muY, sigmaY});

                //action sampling 
                std::normal_distribution<float> dX{muX, sigmaX};
                std::normal_distribution<float> dY{muY, sigmaY}; 
                
                rawActionX = dX(gen);
                rawActionY = dY(gen); 
                
                //clamp actions to domain of [-5, 5]
                actionX = 5*std::tanh(rawActionX);
                actionY = 5*std::tanh(rawActionY);

                //log raw actions sampled from distribution
                rawActions.push_back({rawActionX, rawActionY});
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

            //conver to euler angles
            psi = quaternionToEuler(stateQ); 
        }

        //REINFORCE
        for(int i = 0; i < numIterations; i++){
            totalReturn = 0.0;
            for(int k = i; k < (int)reward.size(); k++){
                totalReturn += std::pow(gamma, k-i) * reward[k];
            }
            
            std::array<float, 4> mk = mKTerms(outputs[i], rawActions[i]);
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

            updateParameters(parameters, w1, b1, w2, b2, w3, b3);
        }
        accumlatedReturn += totalReturn;

        //simple logging
        if(e % 10000 == 0 && e > 0){
            std::cout << "AVERAGE RETURN AFTER " << e << " EPISODES = " << (accumlatedReturn / e*1.0) << std::endl;
            std::cout << "FLIGHT TIME AFTER " << e << " EPISODES = " << numIterations*controlDt << "s" << std::endl;
            std::cout << "*************************************************************" << std::endl;
        }
    }

    saveParameters(w1, w2, w3, b1, b2, b3);
    return 0;
}