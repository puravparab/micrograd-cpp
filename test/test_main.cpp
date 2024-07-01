#include <gtest/gtest.h>

// Include test files from engine/
#include "engine/value_test.cpp"

// Include test files from nn/


int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv); // Initialize Google Test
	return RUN_ALL_TESTS(); // Run all tests
}