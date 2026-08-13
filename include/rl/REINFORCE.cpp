#include "REINFORCE.h"

//Gradient term REINFORCE
std::array<float, 4740> gradientTerm(const std::array<float, 4740>& gradX, const std::array<float, 4740>& gradY, float alpha, float G){
    std::array<float, 4740> gradTerm;

    for(int i = 0; i < 4740; i++){
        gradTerm[i] = alpha*G*(gradX[i]+gradY[i]);
    }

    return gradTerm;
}


//REINFORCE parameter update
void REINFORCEupdate(std::array<float, 4740>& parameters, const std::array<float, 4740>& gradientTerm){
    for(int i = 0; i < 4740; i++){
        parameters[i] += gradientTerm[i];

    }
}