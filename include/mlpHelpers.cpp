#include "mlpHelpers.h" 

void initWeightsAndBiases(std::array<float, 64*4>& w1, std::array<float, 64*64>& w2, std::array<float, 4*64>& w3, std::array<float, 64>& b1, std::array<float, 64>& b2, std::array<float, 4>& b3){ 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    //init weights
    for(int i = 0; i < 64*4; i++){
        w1[i] = dist(gen);
        w3[i] = dist(gen);
    }
    for(int i = 0; i < 64*64; i++){
        w2[i] = dist(gen);
    }

    //init biases 
    b1.fill(0.0f);
    b2.fill(0.0f);
    b3.fill(0.0f);
}
float ReLU(float y){
    if(y > 0) return y;
    else return 0.0f;
}
