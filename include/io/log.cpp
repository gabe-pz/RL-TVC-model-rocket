#include "log.h"

void initCSV(const std::string& filename, int code){
    std::ofstream file(filename); 
    if(code == 0){
        if (file.is_open()) {
            file << "t,rotX\n"; //Write header
        }
    }
    else if(code == 1){
        if (file.is_open()) {
            file << "t,rotY\n"; //Write header
        }
    }
    else{
        if (file.is_open()) {
            file << "episodes,averageReturn\n"; //Write header
        }
    }
}

void logToCSV(double x, double y, const std::string& filename){
    std::ofstream file(filename, std::ios::app); //Append mode
    if (file.is_open()) {
        file << x << "," << y << "\n";
    }
}