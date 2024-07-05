#include <gtest/gtest.h>
#include "../../src/engine.h"
#include <vector>

/*
	MULTIPLICATION
*/
// Test Multiplication with two Value instances
TEST(ValueMultiplicationTest, ValueValueMultiplication) {
	Value v1(2.0f);
	Value v2(-3.3f);
	Value v3 = v1 * v2;

	EXPECT_FLOAT_EQ(v1.data, 2.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);
	EXPECT_FLOAT_EQ(v2.data, -3.3f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, -6.6f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);
	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[0], &v1);
	EXPECT_EQ(prev[1], &v2);
}

// Test Multiplication with Value instance and float
TEST(ValueMultiplicationTest, ValueFloatMultiplication) {
	Value v1(2.0f);
	float v2 = -3.3f;
	Value v3 = v1 * v2;

	EXPECT_FLOAT_EQ(v1.data, 2.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, -6.6f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[0], &v1);
	EXPECT_FLOAT_EQ(prev[1]->data, -3.3f);
	EXPECT_FLOAT_EQ(prev[1]->grad, 0.0f);
}

// Test Multiplication with float and Value instance
TEST(ValueMultiplicationTest, FloatValueMultiplication) {
	float v1 = 2.0f;
	Value v2(-3.3f);
	Value v3 = v1 * v2;

	EXPECT_FLOAT_EQ(v2.data, -3.3f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, -6.6f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[1], &v2);
	EXPECT_FLOAT_EQ(prev[0]->data, 2.0f);
	EXPECT_FLOAT_EQ(prev[0]->grad, 0.0f);
}