#!/bin/bash

mkdir -p build

# Compile the tests
g++ -std=c++14 -Ilib/googletest/googletest/include test/test_main.cpp src/engine.cpp lib/libgtest.a -lpthread -o build/run_tests

# Check if compilation was successful
if [ $? -ne 0 ]; then
	echo "Error: Compilation of tests failed"
	exit 1
fi

# Run the tests
./build/run_tests