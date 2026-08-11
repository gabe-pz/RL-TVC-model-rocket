#include "mlp.h" 

float ReLU(float y){
    if(y > 0) return y;
    else return 0.0f;
}

void initWeightsAndBiases(std::array<std::array<float, 4>, 64>& w1, std::array<std::array<float, 64>, 64>& w2, std::array<std::array<float, 4>, 64>& w3, std::array<float, 64>& b1, std::array<float, 64>& b2, std::array<float, 4>& b3){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    //init weights
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 4; j++){
            w1.at(i).at(j) = dist(gen);
            w3.at(i).at(j) = dist(gen);
        }
    }
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 64; j++){
            w1.at(i).at(j) = dist(gen);
            w3.at(i).at(j) = dist(gen);
        }
    }
    
    //init biases 
    b1.fill(0.0f);
    b2.fill(0.0f);
    b3.fill(0.0f);
}


//*****Pre-activations and activations*****
template<typename T, std::size_t in, std::size_t out>
std::array<float, out> preActivations(const std::array<std::array<float, in>, out>& w, const std::array<float, out>& b, const std::array<T, in>& vecIn){
    std::array<float, out> y{};
    
    //pre-activations no bias
    for(int i = 0; i < out; i++){
        for(int j = 0; j < in; j++){
            y.at(i) += w.at(i).at(j)*vecIn.at(j);
        }
    }

    //add bias
    for(int i = 0; i < out; i++){
        y.at(i) += b.at(i);
    }

    return y;
}

template<std::size_t out>
std::array<float, out> activations(std::array<float, out> y){
    std::array<float, out> a;

    for(int i = 0; i< out; i++){
        a.at(i) = ReLU(y.at(i));
    }

    return a;
}

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
float sumDerivatives(int k, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3, std::array<float, 64> y2){
    float partialN = 0; 

    for(int n = 0; n < 64; n++){
        partialN += derivativePreactivationWrtActivation(w3, k, n)*derivativeActivationWrtPreactivation(y2.at(n))*derivativePreactivationWrtActivation(w2, k, n);
    }
}

float derivativeZkWrtW_1ij(const std::array<double, 4>& stateVec, std::array<float, 64> y1, std::array<float, 4> outputs, int i, int j, int k){
    float p1 = derivativeActivationWrtPreactivation(outputs.at(k));
    float p2 = derivativeActivationWrtPreactivation(y1.at(i));
    float p3 = derivativePreactivationWrtWeight(stateVec, j);
    return p1*p2*p3;
}
float derivativeZkWrtB_1ij(std::array<float, 64> y1, std::array<float, 4> outputs, int i, int k){
    float p1 = derivativeActivationWrtPreactivation(outputs.at(k));
    float p2 = derivativeActivationWrtPreactivation(y1.at(i));
    return p1*p2*1.0f;
}

std::array<std::array<float, 4>, 2> mKAndPartialSummed(const std::array<float, 4>& outputs, const std::array<float, 64>& y2, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 4>, 64>& w3, const std::array<float, 2>& actions){
    std::array<std::array<float, 4>, 2> output;

    std::array<float, 4> mK;
    std::array<float, 4> partialsN;

    for(int n = 0; n < 4; n++){
        partialsN.at(n) = sumDerivatives(n, w2, w3, y2);
    }   
    for(int i = 0; i < 2; i++){
        for(int k = 0; k < 4; k += 2){
            mK.at(k) = (actions.at(i) - outputs.at(k)) / ((outputs.at(k+1)*outputs.at(k+1))*1.0f);
            mK.at(k+1) = ((actions.at(i)-outputs.at(k)*actions.at(i)-outputs.at(k)) - (outputs.at(k+1)*outputs.at(k+1))) / (outputs.at(k+1)*outputs.at(k+1)*outputs.at(k+1));
        }
    }
    output.at(0) = mK;
    output.at(1) = partialsN;
    return output;
}

//*****GRADIENTS*****

//part 1 of grads of log of policy wrt to weights in first layer
std::array<std::array<float, 64*4>, 2> gradientLogPoliciesW1(const std::array<double, 4>& stateVec, std::array<float, 64>& y1, const std::array<float, 4>& outputs, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN){
    std::array<std::array<float, 64*4>, 2> grads; 
    std::array<float, 64*4> gradientLogPolicyXW1;
    std::array<float, 64*4> gradientLogPolicyYW1;


    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 4; j++){
            gradientLogPolicyXW1.at(i) = mK.at(0)*derivativeZkWrtW_1ij(stateVec, y1, outputs, i, j, 0)*partialsN.at(0) + mK.at(1)*derivativeZkWrtW_1ij(stateVec, y1, outputs, i, j, 1)*partialsN.at(1);
        }
    }

    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 4; j++){
            gradientLogPolicyYW1.at(i) = mK.at(2)*derivativeZkWrtW_1ij(stateVec, y1, outputs, i, j, 2)*partialsN.at(2) + mK.at(3)*derivativeZkWrtW_1ij(stateVec, y1, outputs, i, j, 3)*partialsN.at(3);
        }
    }

    
    grads.at(0) = gradientLogPolicyXW1;
    grads.at(1) = gradientLogPolicyYW1;


    return grads; 
}

//part 2 of grads of log of policy wrt to biases in first layer
std::array<std::array<float, 64>, 2> gradientLogPoliciesB1(const std::array<float, 64>& y1, const std::array<float, 4>& outputs, const std::array<float, 4>& mK, const std::array<float, 4>& partialsN){
    std::array<std::array<float, 64>, 2> grads; 
    std::array<float, 64> gradientLogPolicyXB1;
    std::array<float, 64> gradientLogPolicyYB1; 

    for(int i = 0; i < 64; i++){
        gradientLogPolicyXB1.at(i) = mK.at(0)*derivativeZkWrtB_1ij(y1, outputs, i, 0)*partialsN.at(0) + mK.at(1)*derivativeZkWrtB_1ij(y1, outputs, i, 1)*partialsN.at(1);
    }

    for(int i = 0; i < 64; i++){
        gradientLogPolicyYB1.at(i) = mK.at(2)*derivativeZkWrtB_1ij(y1, outputs, i, 2)*partialsN.at(2) + mK.at(3)*derivativeZkWrtB_1ij(y1, outputs, i, 3)*partialsN.at(3);
    }

    
    grads.at(0) = gradientLogPolicyXB1;
    grads.at(1) = gradientLogPolicyYB1;


    return grads; 
}