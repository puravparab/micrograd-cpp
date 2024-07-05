#include <gtest/gtest.h>
#include "../../src/engine.h"
#include <vector>

/*
	ADDITION
*/
// Test Addition with two Value instances
TEST(ValueAdditionTest, ValueValueAddition) {
	Value v1(1.0f);
	Value v2(3.3f);
	Value v3 = v1 + v2;

	EXPECT_FLOAT_EQ(v1.data, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);
	EXPECT_FLOAT_EQ(v2.data, 3.3f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, 4.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);
	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[0], &v1);
	EXPECT_EQ(prev[1], &v2);
}

// Test Addition with Value instance and float
TEST(ValueAdditionTest, ValueFloatAddition) {
	Value v1(1.0f);
	float v2 = 3.3f;
	Value v3 = v1 + v2;

	EXPECT_FLOAT_EQ(v1.data, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, 4.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[0], &v1);
	EXPECT_FLOAT_EQ(prev[1]->data, 3.3f);
	EXPECT_FLOAT_EQ(prev[1]->grad, 0.0f);
}

// Test Addition with float and Value instance
TEST(ValueAdditionTest, FloatValueAddition) {
	float v1 = 1.0f;
	Value v2(3.3f);
	Value v3 = v1 + v2;

	EXPECT_FLOAT_EQ(v2.data, 3.3f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, 4.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[1], &v2);
	EXPECT_FLOAT_EQ(prev[0]->data, 1.0f);
	EXPECT_FLOAT_EQ(prev[0]->grad, 0.0f);
}


/*
	SUBTRACTION
*/
// Test Subtraction with two Value instances
TEST(ValueSubtractionTest, ValueValueSubtraction) {
	Value v1(1.0f);
	Value v2(-3.3f);
	Value v3 = v1 + v2;

	EXPECT_FLOAT_EQ(v1.data, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);
	EXPECT_FLOAT_EQ(v2.data, -3.3f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, -2.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);
	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[0], &v1);
	EXPECT_EQ(prev[1], &v2);
}

// Test Subtraction with Value instance and float
TEST(ValueSubtractionTest, ValueFloatSubtraction) {
	Value v1(1.0f);
	float v2 = -3.3f;
	Value v3 = v1 + v2;

	EXPECT_FLOAT_EQ(v1.data, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, -2.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[0], &v1);
	EXPECT_FLOAT_EQ(prev[1]->data, -3.3f);
	EXPECT_FLOAT_EQ(prev[1]->grad, 0.0f);
}

// Test Subtraction with float and Value instance
TEST(ValueSubtractionTest, FloatValueSubtraction) {
	float v1 = 1.0f;
	Value v2(-3.3f);
	Value v3 = v1 + v2;

	EXPECT_FLOAT_EQ(v2.data, -3.3f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	EXPECT_FLOAT_EQ(v3.data, -2.3f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	const std::vector<Value*>& prev = v3.getPrev();
	ASSERT_EQ(prev.size(), 2);
	EXPECT_EQ(prev[1], &v2);
	EXPECT_FLOAT_EQ(prev[0]->data, 1.0f);
	EXPECT_FLOAT_EQ(prev[0]->grad, 0.0f);
}