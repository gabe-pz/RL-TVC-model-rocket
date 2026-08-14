#include "saveAndLoad.h"


void saveParameters(const std::array<std::array<float, 4>, 64>& w1, const std::array<std::array<float, 64>, 64>& w2, const std::array<std::array<float, 64>, 4>& w3, 
    const std::array<float, 64>& b1, const std::array<float, 64>& b2, const std::array<float, 4>& b3){
   
    std::ofstream out("../weights.bin", std::ios::binary);
    
    out.write(reinterpret_cast<const char*>(&w1), sizeof(w1));
    out.write(reinterpret_cast<const char*>(&w2), sizeof(w2));
    out.write(reinterpret_cast<const char*>(&w3), sizeof(w3));
    out.write(reinterpret_cast<const char*>(&b1), sizeof(b1));
    out.write(reinterpret_cast<const char*>(&b2), sizeof(b2));
    out.write(reinterpret_cast<const char*>(&b3), sizeof(b3));
    
    out.close();
}

void loadParameters(std::array<std::array<float, 4>, 64>& w1, std::array<std::array<float, 64>, 64>& w2, std::array<std::array<float, 64>, 4>& w3, 
    std::array<float, 64>& b1, std::array<float, 64>& b2, std::array<float, 4>& b3){

        std::ifstream in("../weights.bin", std::ios::binary);
        
        in.read(reinterpret_cast<char*>(&w1), sizeof(w1));
        in.read(reinterpret_cast<char*>(&w2), sizeof(w2));
        in.read(reinterpret_cast<char*>(&w3), sizeof(w3));
        in.read(reinterpret_cast<char*>(&b1), sizeof(b1));
        in.read(reinterpret_cast<char*>(&b2), sizeof(b2));
        in.read(reinterpret_cast<char*>(&b3), sizeof(b3));
        
        in.close();
}