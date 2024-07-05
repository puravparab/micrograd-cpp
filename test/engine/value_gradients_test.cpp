#include <gtest/gtest.h>
#include "../../src/engine.h"
#include <vector>

/*
	ADDITION
*/
// Test gradients after addition
TEST(AdditionGradientsTest, SimpleAdditionBackward) {
	Value v1(1.0f);
	Value v2(3.3f);
	Value v3 = v1 + v2;
	EXPECT_FLOAT_EQ(v3.data, 4.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);
	v3.backward();
	EXPECT_FLOAT_EQ(v3.grad, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 1.0f);
	EXPECT_FLOAT_EQ(v2.grad, 1.0f);
}

/*
	MULTIPLICATION
*/
// Test gradients after simple multiplication
TEST(MultiplicationGradientsTest, SimpleMultiplicationBackward) {
	Value v1 = Value(1.0f);
	Value v2 = Value(3.3f);
	Value v3 = v1 * v2;
	EXPECT_FLOAT_EQ(v3.data, 3.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);
	v3.backward();
	EXPECT_FLOAT_EQ(v3.grad, 1.0f); // v3
	EXPECT_FLOAT_EQ(v1.grad, 3.3f); // v2
	EXPECT_FLOAT_EQ(v2.grad, 1.0f); // v1
}

// Test gradients after chained multiplication
TEST(MultiplicationGradientsTest, ChainedMultiplicationBackward) {
	Value v1 = Value(1.0f);
	Value v2 = Value(3.3f);
	Value v3 = v1 * v2;
	Value v4 = v1 * v3;
	v4.backward();
	EXPECT_FLOAT_EQ(v4.data, 3.3f); // v4
	EXPECT_FLOAT_EQ(v4.grad, 1.0f);
	EXPECT_FLOAT_EQ(v3.data, 3.3f); // v3
	EXPECT_FLOAT_EQ(v3.grad, 1.0f);
	EXPECT_FLOAT_EQ(v2.data, 3.3f); // v2
	EXPECT_FLOAT_EQ(v2.grad, 1.0f);
	EXPECT_FLOAT_EQ(v1.data, 1.0f); // v1
	EXPECT_FLOAT_EQ(v1.grad, 6.6f);
}

/*
	ACTIVATION FUNCTIONS
*/
// Test gradients after TanH
TEST(ActivationsGradientsTest, TanHBackward) {
	Value v1 = Value(10.0);
	Value v2 = v1.tanh();
	EXPECT_FLOAT_EQ(v2.data, 1.0f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	v2.backward();
	EXPECT_FLOAT_EQ(v2.grad, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);
}

// Test gradients after ReLU
TEST(ActivationsGradientsTest, ReLUBackward) {
	Value v1 = Value(-1.0);
	Value v2 = v1.relu();
	EXPECT_FLOAT_EQ(v2.data, 0.0f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	v2.backward();
	EXPECT_FLOAT_EQ(v2.grad, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);

	Value v3 = Value(1.0);
	Value v4 = v3.relu();
	EXPECT_FLOAT_EQ(v4.data, 1.0f);
	EXPECT_FLOAT_EQ(v4.grad, 0.0f);
	v4.backward();
	EXPECT_FLOAT_EQ(v4.grad, 1.0f);
	EXPECT_FLOAT_EQ(v3.grad, 1.0f);
}

// Test gradients after Sigmoid
TEST(ActivationsGradientsTest, SigmoidBackward) {
	Value v1 = Value(0.0);
	Value v2 = v1.sigmoid();
	EXPECT_FLOAT_EQ(v2.data, 0.5f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	v2.backward();
	EXPECT_FLOAT_EQ(v2.grad, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.25f);
}