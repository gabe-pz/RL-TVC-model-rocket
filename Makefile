CXX = g++

.PHONY: sim train clean

#Build & run rlSim
sim:
	mkdir -p bin
	$(CXX) -Wall -g -Iinclude src/rlSim.cpp $(shell find include -name '*.cpp') -o bin/rlSim -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
	./bin/rlSim

#Build & run train
train:
	mkdir -p bin
	$(CXX) -Wall -O2 -std=c++17 src/train.cpp $(filter-out include/graphics/%.cpp, $(wildcard include/*/*.cpp)) -o bin/train
	./bin/train

clean:
	rm -rf bin