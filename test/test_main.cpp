#include <gtest/gtest.h>

// test files from engine/
#include "engine/value_test.cpp"
#include "engine/value_addition_test.cpp"
#include "engine/value_multiplication_test.cpp"
#include "engine/value_activations_test.cpp"
#include "engine/value_gradients_test.cpp"

// test files from nn/
#include "nn/nn_test.cpp"

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv); // Initialize Google Test
	return RUN_ALL_TESTS(); // Run all tests
}