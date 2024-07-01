#!/bin/bash

cd lib/googletest
mkdir -p build
cd build

# Configure and build Google Test
g++ -std=c++14 -isystem ../googletest/include -I../googletest -pthread -c ../googletest/src/gtest-all.cc
ar -rv libgtest.a gtest-all.o

# Move the built library to the project's lib directory
mv libgtest.a ../../

# Clean up
cd ../..
rm -rf build

echo "Google Test has been built and the library is placed in the lib directory."