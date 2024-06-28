#!/bin/bash

g++ -o micrograd main.cpp engine.cpp nn.cpp -std=c++11
# Check if compilation was successful
if [ $? -ne 0 ]; then
	echo "Error: Compilation of main program failed"
	exit 1
fi

./micrograd