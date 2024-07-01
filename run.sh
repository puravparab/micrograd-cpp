#!/bin/bash

mkdir -p build

# compile src
g++ -std=c++14 -o build/micrograd main.cpp engine.cpp nn.cpp

# Check if compilation was successful
if [ $? -ne 0 ]; then
	echo "Error: Compilation of main program failed"
	exit 1
fi

./build/micrograd