#include <gtest/gtest.h>
#include "../../src/engine.h"
#include <vector>

/*
	TanH
*/
// Test Tanh activation function
TEST(ValueActivationsTest, ValueTanH) {
	Value v1 = Value(10.0);
	Value v2 = v1.tanh();
	EXPECT_FLOAT_EQ(v2.data, 1.0f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev1= v2.getPrev();
	ASSERT_EQ(prev1.size(), 1);
	EXPECT_EQ(prev1[0], &v1);

	v1 = Value(-10.0);
	v2 = v1.tanh();
	EXPECT_FLOAT_EQ(v2.data, -1.0f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev2 = v2.getPrev();
	ASSERT_EQ(prev2.size(), 1);
	EXPECT_EQ(prev2[0], &v1);

	v1 = Value(1.0);
	v2 = v1.tanh();
	EXPECT_FLOAT_EQ(v2.data, 0.7615942f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev3 = v2.getPrev();
	ASSERT_EQ(prev3.size(), 1);
	EXPECT_EQ(prev3[0], &v1);

	v1 = Value(-1.0);
	v2 = v1.tanh();
	EXPECT_FLOAT_EQ(v2.data, -0.7615942f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev4 = v2.getPrev();
	ASSERT_EQ(prev4.size(), 1);
	EXPECT_EQ(prev4[0], &v1);
}

/*
	ReLU
*/
// Test ReLU activation function
TEST(ValueActivationsTest, ValueReLU) {
	Value v1 = Value(1.0);
	Value v2 = v1.relu();
	EXPECT_FLOAT_EQ(v2.data, 1.0f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev1= v2.getPrev();
	ASSERT_EQ(prev1.size(), 1);
	EXPECT_EQ(prev1[0], &v1);

	v1 = Value(0.0);
	v2 = v1.relu();
	EXPECT_FLOAT_EQ(v2.data, 0.0f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev2 = v2.getPrev();
	ASSERT_EQ(prev2.size(), 1);
	EXPECT_EQ(prev2[0], &v1);

	v1 = Value(-1.0);
	v2 = v1.relu();
	EXPECT_FLOAT_EQ(v2.data, 0.0);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev3 = v2.getPrev();
	ASSERT_EQ(prev3.size(), 1);
	EXPECT_EQ(prev3[0], &v1);
}

/*
	Sigmoid
*/
// Test Sigmoid activation function
TEST(ValueActivationsTest, ValueSigmoid) {
	Value v1 = Value(0.0);
	Value v2 = v1.sigmoid();
	EXPECT_FLOAT_EQ(v2.data, 0.5f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev1= v2.getPrev();
	ASSERT_EQ(prev1.size(), 1);
	EXPECT_EQ(prev1[0], &v1);

	v1 = Value(2);
	v2 = v1.sigmoid();
	EXPECT_FLOAT_EQ(v2.data, 0.8807971f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev2 = v2.getPrev();
	ASSERT_EQ(prev2.size(), 1);
	EXPECT_EQ(prev2[0], &v1);

	v1 = Value(-2.0);
	v2 = v1.sigmoid();
	EXPECT_FLOAT_EQ(v2.data, 0.1192029f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	const std::vector<Value*>& prev3 = v2.getPrev();
	ASSERT_EQ(prev3.size(), 1);
	EXPECT_EQ(prev3[0], &v1);
}