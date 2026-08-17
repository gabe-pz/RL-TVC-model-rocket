#include "REINFORCE.h"



//Gradient term in REINFORCE
std::array<float, 4740> gradientTermREINFORCE(const std::array<float, 4740>& gradX, const std::array<float, 4740>& gradY, float alpha, float G){
    std::array<float, 4740> gradTerm;
    float maxStep = 1.0;
    for(int i = 0; i < 4740; i++){
        gradTerm[i] = alpha*std::clamp(G*(gradX[i]+gradY[i]), -maxStep, maxStep); 
    }

    return gradTerm;
}

//turn weights and bias into single vector, theta
std::array<float, 4740> flattenParameters(const std::array<std::array<float, 4>, 64>& w1, const std::array<float, 64>& b1, const std::array<std::array<float, 64>, 64>& w2, const std::array<float, 64>& b2,
    const std::array<std::array<float, 64>, 4>& w3, const std::array<float, 4>& b3){

    std::array<float, 4740> output;
    int index = 0;

    //w1
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 4; j++){
            output[index] = w1[i][j];
            index++;
        }
    }

    //b1
    for(int i = 0; i < 64; i++){
        output[index] = b1[i];
        index++;
    }

    //w2
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 64; j++){
            output[index] = w2[i][j];
            index++;
        }
    }

    //b2
    for(int i = 0; i < 64; i++){
        output[index] = b2[i];
        index++;
    }

    //w3
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 64; j++){
            output[index] = w3[i][j];
            index++;
        }
    }

    //b3
    for(int i = 0; i < 4; i++){
        output[index] = b3[i];
        index++;
    }

    return output;
} 

//apply the updated vector to each weigth and bias
void updateParameters(const std::array<float, 4740>& flattened, std::array<std::array<float, 4>, 64>& w1, std::array<float, 64>& b1, std::array<std::array<float, 64>, 64>& w2, std::array<float, 64>& b2, 
    std::array<std::array<float, 64>, 4>& w3, std::array<float, 4>& b3){
        int index = 0;
        //w1
        for(int i = 0; i < 64; i++){
            for(int j = 0; j < 4; j++){
                w1[i][j] = flattened[index];
                index++;
            }
        }
        //b1
        for(int i = 0; i < 64; i++){
            b1[i] = flattened[index];
            index++;
        }
        //w2
        for(int i = 0; i < 64; i++){
            for(int j = 0; j < 64; j++){
                w2[i][j] = flattened[index];
                index++;
            }
        }
        //b2
        for(int i = 0; i < 64; i++){
            b2[i] = flattened[index];
            index++;
        }
        //w3
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 64; j++){
                w3[i][j] = flattened[index];
                index++;
            }
        }
        //b3
        for(int i = 0; i < 4; i++){
            b3[i] = flattened[index];
            index++;
        }
} 


//REINFORCE parameter update
void REINFORCEupdate(std::array<std::array<float, 4>, 64>& w1, std::array<float, 64>& b1, std::array<std::array<float, 64>, 64>& w2, std::array<float, 64>& b2, 
    std::array<std::array<float, 64>, 4>& w3, std::array<float, 4>& b3, const std::array<float, 4740>& gradientTerm){

    //create parameter vector
    std::array<float, 4740> parameterVector = flattenParameters(w1, b1, w2, b2, w3, b3);
    

    //apply update to parameter vector
    for(int i = 0; i < 4740; i++){
        parameterVector[i] += gradientTerm[i];
    }


    //apply updated vector to each weight and bias
    updateParameters(parameterVector, w1, b1, w2, b2, w3, b3);
}