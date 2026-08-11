#include "gradient.h"


//*****DERIVATIVES*****
float derivativeActivationWrtPreactivation(float yLk){
    if(yLk > 0) return 1.0f;
    else return 0.0f; 
}

template<std::size_t in, std::size_t out>
float derivativePreactivationWrtActivation(const std::array<std::array<float, in>, out>& w, int n, int m){
    return w.at(n).at(m);
}

template<typename T, std::size_t in>
float derivativePreactivationWrtWeight(const std::array<T, in>& vecIn, int j){
    return vecIn.at(j)*1.0f;
} 


//*****HELPER DERIVATIVE FUNCTIONS*****
//sum term in expression for derivatives for z_k wrt weights and biases in first layer
float sumDerivatives(int k, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3, std::array<float, 64> y2){
    float partialN = 0; 

    for(int n = 0; n < 64; n++){
        partialN += derivativePreactivationWrtActivation(w3, k, n)*derivativeActivationWrtPreactivation(y2.at(n))*derivativePreactivationWrtActivation(w2, k, n);
    }
}

//helper function that returns the common terms in each gradient componenet
std::array<float, 4> mKTerms(const std::array<float, 4>& outputs,  const std::array<float, 2>& actions){
    std::array<float, 4> mK;

    for(int i = 0; i < 2; i++){
        for(int k = 0; k < 4; k += 2){
            mK.at(k) = (actions.at(i) - outputs.at(k)) / ((outputs.at(k+1)*outputs.at(k+1))*1.0f);
            mK.at(k+1) = ((actions.at(i)-outputs.at(k)*actions.at(i)-outputs.at(k)) - (outputs.at(k+1)*outputs.at(k+1))) / (outputs.at(k+1)*outputs.at(k+1)*outputs.at(k+1));
        }
    }

    return mK;
}

//helper function that returns the sum of the partials of the product
std::array<float, 4> partialsSummed(const std::array<float, 64>& y2, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3){
    std::array<float, 4> partialsN;

    for(int n = 0; n < 4; n++){
        partialsN.at(n) = sumDerivatives(n, w2, w3, y2);
    }   

    return partialsN;
}

//The product of the three derivatives in the expression for the derivatives of z_k wrt weights and biases in 1st layer
float derivativeZkWrtW_1ij(const std::array<double, 4>& stateVec, const std::array<float, 64>& y1, const std::array<float, 4>& yOut, int i, int j, int k){
    float p1 = derivativeActivationWrtPreactivation(yOut.at(k));
    float p2 = derivativeActivationWrtPreactivation(y1.at(i));
    float p3 = derivativePreactivationWrtWeight(stateVec, j);
    return p1*p2*p3;
}
float derivativeZkWrtB_1ij(std::array<float, 64> y1, std::array<float, 4> yOut, int i, int k){
    float p1 = derivativeActivationWrtPreactivation(yOut.at(k));
    float p2 = derivativeActivationWrtPreactivation(y1.at(i));
    return p1*p2*1.0f;
}

//expressions for derivative of z_k wrt weights and biases in 2nd layer
float derivativeZkWrtW_2ij(const std::array<float, 64>& y2, const std::array<float, 4>& yOut, const std::array<float, 64> a1, const std::array<std::array<float, 4>, 64>& w3, int i, int j, int k){
    float p1 = derivativeActivationWrtPreactivation(yOut.at(k));
    float p2 = derivativePreactivationWrtActivation(w3, k, i);
    float p3 = derivativeActivationWrtPreactivation(y2.at(i));
    float p4 = derivativePreactivationWrtWeight(a1, j);
    return p1*p2*p3*p4;
}
float derivativeZkWrtB_2ij(const std::array<float, 64>& y2, const std::array<float, 4>& yOut, const std::array<std::array<float, 4>, 64>& w3, int i, int k){
    float p1 = derivativeActivationWrtPreactivation(yOut.at(k));
    float p2 = derivativePreactivationWrtActivation(w3, k, i);
    float p3 = derivativeActivationWrtPreactivation(y2.at(i));
    float p4 = 1.0f;
    return p1*p2*p3*p4;
}

//expressions for derivative of z_k wrt weights and biases in 3rd layer
float derivativeZkWrtW_3ij(const std::array<float, 4>& yOut, const std::array<float, 64> a2, int j, int k){
    float p1 = derivativeActivationWrtPreactivation(yOut.at(k));
    float p2 = derivativePreactivationWrtWeight(a2, j);
    return p1*p2;
}
float derivativeZkWrtB_3ij(const std::array<float, 4>& yOut, int k){
    float p1 = derivativeActivationWrtPreactivation(yOut.at(k));
    return p1*1.0f;
}


//*****GRADIENTS*****
//part 1 of gradients of log of the policies wrt to weights in first layer
std::array<std::array<float, 64*4>, 2> gradientLogPoliciesW1(const std::array<double, 4>& stateVec, std::array<float, 64>& y1, const std::array<float, 4>& yOut, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN){
    std::array<std::array<float, 64*4>, 2> gradients; 
    std::array<float, 64*4> gradientLogPolicyXW1;
    std::array<float, 64*4> gradientLogPolicyYW1;


    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 4; j++){
            gradientLogPolicyXW1.at(i*4+j) = mK.at(0)*derivativeZkWrtW_1ij(stateVec, y1, yOut, i, j, 0)*partialsN.at(0) + mK.at(1)*derivativeZkWrtW_1ij(stateVec, y1, yOut, i, j, 1)*partialsN.at(1);
        }
    }

    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 4; j++){
            gradientLogPolicyYW1.at(i*4+j) = mK.at(2)*derivativeZkWrtW_1ij(stateVec, y1, yOut, i, j, 2)*partialsN.at(2) + mK.at(3)*derivativeZkWrtW_1ij(stateVec, y1, yOut, i, j, 3)*partialsN.at(3);
        }
    }

    
    gradients.at(0) = gradientLogPolicyXW1;
    gradients.at(1) = gradientLogPolicyYW1;


    return gradients; 
}
//part 2 of gradients of log of the policies wrt to biases in first layer
std::array<std::array<float, 64>, 2> gradientLogPoliciesB1(const std::array<float, 64>& y1, const std::array<float, 4>& outputs, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN){
    std::array<std::array<float, 64>, 2> gradients; 
    std::array<float, 64> gradientLogPolicyXB1;
    std::array<float, 64> gradientLogPolicyYB1; 

    for(int i = 0; i < 64; i++){
        gradientLogPolicyXB1.at(i) = mK.at(0)*derivativeZkWrtB_1ij(y1, outputs, i, 0)*partialsN.at(0) + mK.at(1)*derivativeZkWrtB_1ij(y1, outputs, i, 1)*partialsN.at(1);
    }

    for(int i = 0; i < 64; i++){
        gradientLogPolicyYB1.at(i) = mK.at(2)*derivativeZkWrtB_1ij(y1, outputs, i, 2)*partialsN.at(2) + mK.at(3)*derivativeZkWrtB_1ij(y1, outputs, i, 3)*partialsN.at(3);
    }

    
    gradients.at(0) = gradientLogPolicyXB1;
    gradients.at(1) = gradientLogPolicyYB1;


    return gradients; 
}

//part 3 of gradients of log of the policies wrt to weights in the second layer
std::array<std::array<float, 64*64>, 2> gradientLogPoliciesW2(const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 64> a1, const std::array<float, 64> y2, const std::array<float, 4> yOut, const std::array<float, 4>& mK){
    std::array<std::array<float, 64*64>, 2> outputs;

    std::array<float, 64*64> gradientLogPolicyXW2; 
    std::array<float, 64*64> gradientLogPolicyYW2; 

    //x
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 64; j++){
            gradientLogPolicyXW2.at(i*64+j) = mK.at(0)*derivativeZkWrtW_2ij(y2, yOut, a1, w3, i, j, 0) + mK.at(1)*derivativeZkWrtW_2ij(y2, yOut, a1, w3, i, j, 1); 
        }
    }
    
    //y
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 64; j++){
            gradientLogPolicyYW2.at(i*64+j) = mK.at(2)*derivativeZkWrtW_2ij(y2, yOut, a1, w3, i, j, 2) + mK.at(3)*derivativeZkWrtW_2ij(y2, yOut, a1, w3, i, j, 3); 
        }
    }

    outputs.at(0) = gradientLogPolicyXW2;
    outputs.at(1) = gradientLogPolicyYW2;

    return outputs;
}
//part 4 of gradients of log of the policies wrt to biases in the second layer
std::array<std::array<float, 64>, 2> gradientLogPoliciesB2(const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 64> y2, const std::array<float, 4> yOut, const std::array<float, 4>& mK){
    std::array<std::array<float, 64>, 2> outputs;

    std::array<float, 64> gradientLogPolicyXB2; 
    std::array<float, 64> gradientLogPolicyYB2; 

    //x
    for(int i = 0; i < 64; i++){
        gradientLogPolicyXB2.at(i) = mK.at(0)*derivativeZkWrtB_2ij(y2, yOut, w3, i, 0) + mK.at(1)*derivativeZkWrtB_2ij(y2, yOut, w3, i, 1);
    }
    
    //y
    for(int i = 0; i < 64; i++){
        gradientLogPolicyYB2.at(i) = mK.at(2)*derivativeZkWrtB_2ij(y2, yOut, w3, i, 2) + mK.at(3)*derivativeZkWrtB_2ij(y2, yOut, w3, i, 3);
    }

    outputs.at(0) = gradientLogPolicyXB2;
    outputs.at(1) = gradientLogPolicyYB2;

    return outputs;
}


//part 5 of gradients of log of the policies wrt to weights in the third layer
std::array<std::array<float, 4*64>, 2> gradientLogPoliciesW3(const std::array<float, 4>& yOut, const std::array<float, 64> a2, const std::array<float, 4>& mK){
    std::array<std::array<float, 4*64>, 2> outputs;

    std::array<float, 4*64> gradientLogPolicyXW3;
    std::array<float, 4*64> gradientLogPolicyYW3;

    //x
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 64; j++){
            gradientLogPolicyXW3.at(i*64+j) = mK.at(0)*derivativeZkWrtW_3ij(yOut, a2, j, 0)+mK.at(1)*derivativeZkWrtW_3ij(yOut, a2, j, 1);
        }
    }


    //y
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 64; j++){
            gradientLogPolicyYW3.at(i*64+j) = mK.at(2)*derivativeZkWrtW_3ij(yOut, a2, j, 2)+mK.at(3)*derivativeZkWrtW_3ij(yOut, a2, j, 3);
        }
    }

    outputs.at(0) = gradientLogPolicyXW3;
    outputs.at(1) = gradientLogPolicyYW3;

    return outputs;
}
//part 6 of gradients of log of the policies wrt to biases in the third layer
std::array<std::array<float, 4>, 2> gradientLogPoliciesB3(const std::array<float, 4>& yOut, const std::array<float, 4>& mK){
    std::array<std::array<float, 4>, 2> outputs;

    std::array<float, 4> gradientLogPolicyXB3;
    std::array<float, 4> gradientLogPolicyYB3;

    //x
    for(int i = 0; i < 4; i++){
        gradientLogPolicyXB3.at(i) = mK.at(0)*derivativeZkWrtB_3ij(yOut, 0) + mK.at(1)*derivativeZkWrtB_3ij(yOut, 1); 
    }


    //x
    for(int i = 0; i < 4; i++){
        gradientLogPolicyYB3.at(i) = mK.at(2)*derivativeZkWrtB_3ij(yOut, 2) + mK.at(3)*derivativeZkWrtB_3ij(yOut, 3); 
    }

    outputs.at(0) = gradientLogPolicyXB3;
    outputs.at(1) = gradientLogPolicyYB3;

    return outputs;
}

//construction function
std::array<std::array<float, 4740>, 2> constructGradients(const std::array<std::array<float, 64*4>, 2>& gW1, const std::array<std::array<float, 64>, 2>& gB1, const std::array<std::array<float, 64*64>, 2>& gW2, 
    const std::array<std::array<float, 64>, 2>& gB2, const std::array<std::array<float, 4*64>, 2>& gW3, const std::array<std::array<float, 4>, 2>& gB3){

        std::array<std::array<float, 4740>, 2> outputs; 

        for(int i = 0; i < 2; i++){
            int index = 0; 
            
            for(int j = 0; j < 64*4; j++){
                outputs[i][index++] = gW1[i][j];
            }
            for(int j = 0; j < 64; j++){
                outputs[i][index++] = gB1[i][j];
            }
            for(int j = 0; j < 64*64; j++){
                outputs[i][index++] = gW2[i][j];
            }
            for(int j = 0; j < 64; j++){
                outputs[i][index++] = gB2[i][j];
            }
            for(int j = 0; j < 4*64; j++){
                outputs[i][index++] = gW3[i][j];
            }
            for(int j = 0; j < 4; j++){
                outputs[i][index++] = gB3[i][j];
            }

        }

        return outputs;
}
