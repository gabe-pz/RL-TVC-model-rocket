#include "mlp.h" 

float ReLU(float y){
    if(y > 0) return y;
    else return 0.0f;
}

//*****FUNCTIONS FOR PARAMETERS*****
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

std::array<float, 4740> flatten_parameters(const std::array<std::array<float, 4>, 64>& w1, const std::array<float, 64>& b1, const std::array<std::array<float, 64>, 64>& w2, const std::array<float, 64>& b2,
    const std::array<std::array<float, 4>, 64>& w3,
    const std::array<float, 4>& b3)
{

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
    for(int i = 0; i < 64; i++){
        for(int j = 0; j < 4; j++){
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