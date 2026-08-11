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