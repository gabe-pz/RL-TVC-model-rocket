CXX = g++
CXXFLAGS = -Wall -g -std=c++17
TARGET = bin/train
SRC = src/train.cpp $(filter-out include/graphics/%.cpp, $(wildcard include/*/*.cpp))
HDR = $(filter-out include/graphics/%.h, $(wildcard include/*/*.h)) 

.PHONY: run all clean

run: all
	./$(TARGET)

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	mkdir -p bin
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -rf bin