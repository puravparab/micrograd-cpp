#!/bin/bash

mkdir -p build

# compile src
g++ -std=c++14 -o build/micrograd src/main.cpp src/engine.cpp src/nn.cpp

# Check if compilation was successful
if [ $? -ne 0 ]; then
	echo "Error: Compilation of main program failed"
	exit 1
fi

./build/micrograd