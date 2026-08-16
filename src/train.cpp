#include <array>
#include <vector> 
#include <iostream> 
#include <random> 
#include <cmath> 
#include <algorithm> 

#include "../include/rl/mlp.h"
#include "../include/rl/gradient.h"
#include "../include/rl/REINFORCE.h"
#include "../include/io/saveAndLoad.h"
#include "../include/physics/physicsLoop.h"

int main(void){
    //*****RANDOM NUM GEN*****
    std::random_device rd;
    std::mt19937 gen(rd());

    //*****SIMULATION VARS*****
    const double dt = 0.0001;
    const double simTime = tBurn;
    double t = 0.0;
        
    //*******STATE ARRAY INITALIZATIONS*****
    //quaterion initaliztion
    std::array<double, 4> stateQ;
    
    std::array<double, 3> velocity;
    std::array<double, 3> position;

    std::array<double, 3> angularVelocity;
    std::array<double, 2> psi;

    //*****REINFORCEMENT LEARNING*****
    //state vector
    std::vector<std::array<double, 4>> stateVector;
    
    //mlp outputs for training
    MLPoutput mlpOutputTraining;

    //mlp outputs for control
    std::array<float, 4> mlpOutputControl; 
    
    //weights and biases initialization
    std::array<std::array<float, 4>, 64> w1; 
    std::array<std::array<float, 64>, 64> w2; 
    std::array<std::array<float, 64>, 4> w3; 
    std::array<float, 64> b1;
    std::array<float, 64> b2;
    std::array<float, 4> b3;
    initWeightsAndBiases(w1, w2, w3, b1, b2, b3);
        
    //actions
    std::vector<std::array<float, 2>> rawActions;    
    float rawActionX = 0.0f;
    float rawActionY = 0.0f;
    float actionX = 0.0f;
    float actionY = 0.0f;
    
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
    int numEpisodes = 50000;
    int numIterations = 0; 
    int episodesInWindow = 0;
    
    //hyperparameters
    float alpha = 0.00005f;//step size
    float gamma = 0.8f;//discount factor 
    float a = 175.0f;//exp constant
    float b = 5.5f;//angular v penalize factor
    float lowerBoundExp = -1.0f;//the lower bound for clamp of exponential for computing sigma 
    float upperBoundExp = 2.0f;//the upper bound  for clamp of exponential for computing sigma 
    int c = -10;//termination reward


    for(int e = 0; e < numEpisodes; e++){      
        //******WIND AND SERVO OFFSET*****
        
        //wind generation constants per episode
        std::uniform_real_distribution<double> distU(2.0, 8.0); //random average wind speed
        double U = distU(gen);
        
        std::uniform_real_distribution<double> distIntensity(0.10, 0.20); //random turb intensity
        double intensity = distIntensity(gen); 
        
        double sigmaU = intensity * U;

        //random wind generated per episode
        unsigned int episodeSeed = rd();
        int n = (int)(simTime * GEN_FREQ) + 2;
        std::vector<double> pinkU = generatePinkNoise(n, episodeSeed);
        std::vector<double> pinkV = generatePinkNoise(n, episodeSeed + 1);
        std::vector<double> pinkW = generatePinkNoise(n, episodeSeed + 2);
        std::array<std::vector<double>, 3> pinkNoise = {pinkU, pinkV, pinkW};

        //servo offset per episode
        std::uniform_real_distribution<float> distS(0.017f, 0.034f);//misalignment between 1 and 2 deg
        float servosXOffset = distS(gen);
        float servosYOffset = distS(gen);

        //*****RESET*****
        //time and counting
        numIterations = 0;
        t = 0.0;
        timeSinceLastControl = controlDt;

        //state arrays
        stateQ = {1.0, 0.0, 0.0, 0.0};
        
        velocity = {0.0, 0.0, 0.0};
        position = {0.0, 0.0, 0.0};
        
        angularVelocity = {0.0, 0.0, 0.0};
        psi = {0.0, 0.0};
        
        //return
        returnT = 0.0; 

        //storage clearing
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
                        reward.push_back(c);       
                        break;
                    }
                
                    //reward update
                    double rewardT = std::exp(-a*(psi[0]*psi[0]+psi[1]*psi[1])) - b*(angularVelocity[0]*angularVelocity[0]+angularVelocity[1]*angularVelocity[1]);
                    reward.push_back(rewardT);       
                }

                //increment another step
                numIterations ++;

                //log and save state
                std::array<double, 4> state = {psi[0], psi[1], angularVelocity[0], angularVelocity[1]};
                stateVector.push_back(state);
                
                //mlp forward pass 
                mlpOutputControl = mlpControl(state, w1, w2, w3, b1, b2, b3);

                //gaussian distribution parameters
                float muX = mlpOutputControl[0];
                float sigmaX = std::exp(std::clamp(mlpOutputControl[1], lowerBoundExp, upperBoundExp));//ensure that the variance is positive by having network to output log of sigma and thus sigma is exp of that, which will keep it positive
                float muY = mlpOutputControl[2];
                float sigmaY = std::exp(std::clamp(mlpOutputControl[3], lowerBoundExp, upperBoundExp));//also clamping to halt explosions. values using were empirically found

                //gaussian distribution creation
                std::normal_distribution<float> gaussianDx{muX, sigmaX};
                std::normal_distribution<float> gaussianDy{muY, sigmaY}; 
                
                //action sampling
                rawActionX = gaussianDx(gen);
                rawActionY = gaussianDy(gen); 
                
                //clamp actions to domain of [-5 deg, 5 deg], where 0.087 rads is about 5 degs. 
                actionX = 0.087*std::tanh(rawActionX);
                actionY = 0.087*std::tanh(rawActionY);

                //log raw actions sampled from distribution
                rawActions.push_back({rawActionX, rawActionY});
            }

            //******PHYSICS UPDATE******
            physicsUpdate(dt, t, U, sigmaU, actionX, actionY, servosXOffset, servosYOffset, pinkNoise, position, velocity, stateQ, angularVelocity, psi);
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


            //forward pass, get pre-acts and acts, as well as outputs
            mlpOutputTraining = mlpTrain(stateVector[i], w1, w2, w3, b1, b2, b3); 
            
            //pre-activations and activations, comptued for the new weights from the upated theta
            std::array<float, 64> y1 = mlpOutputTraining.y1;
            std::array<float, 64> y2 = mlpOutputTraining.y2;
            std::array<float, 4> y3 = mlpOutputTraining.y3;

            std::array<float, 64> a1 = mlpOutputTraining.a1;
            std::array<float, 64> a2 = mlpOutputTraining.a2;
            std::array<float, 4> a3 = mlpOutputTraining.a3;//(mu_x, log(sigma_x), mu_y, log(sigma_y))^T
            
            //sigmas calculated for mkTerms
            float sigmaX = std::exp(std::clamp(a3[1], lowerBoundExp, upperBoundExp));//same clamping and trick as in physics loop, aka episode generation
            float sigmaY = std::exp(std::clamp(a3[3], lowerBoundExp, upperBoundExp));

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
            
            //flatten weights and biases into single vector, theta
            std::array<float, 4740> parameters = flattenParameters(w1, b1, w2, b2, w3, b3);
            
            //apply the reinforce update
            REINFORCEupdate(parameters, gradTerm);
            
            //update each weight and bias with the reinforce update
            updateParameters(parameters, w1, b1, w2, b2, w3, b3);
        }

        //logging
        returnAccumlated += totalReturn;
        accumlatedFlightTime += numIterations*controlDt;
        episodesInWindow ++; 

        //Log to terminal every 1k eps 
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
    return 0;
}
