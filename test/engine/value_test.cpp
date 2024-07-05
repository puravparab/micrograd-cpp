#include <gtest/gtest.h>
#include "../../src/engine.h"
#include <vector>

// Test the constructor with a float only
TEST(ValueConstructorTest, FloatConstructor) {
	Value v1(0.0f);
	EXPECT_FLOAT_EQ(v1.data, 0.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);

	Value v2(5.5f);
	EXPECT_FLOAT_EQ(v2.data, 5.5f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	Value v3(-1.0f);
	EXPECT_FLOAT_EQ(v3.data, -1.0f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	Value v4(3.14f);
	EXPECT_FLOAT_EQ(v4.data, 3.14f);
	EXPECT_FLOAT_EQ(v4.grad, 0.0f);

	Value v5(32.14f);
	EXPECT_FLOAT_EQ(v5.data, 32.14f);
	EXPECT_FLOAT_EQ(v5.grad, 0.0f);
}

// Test the constructor with float and previous Values
TEST(ValueConstructorTest, FloatAndPreviousValuesConstructor) {
	Value* prev1 = new Value(2.0f);
	Value* prev2 = new Value(-1.0f);
	Value* prev3 = new Value(0.5f);

	// Test with empty previous values
	std::vector<Value*> parents = {};
	Value v1(1.0f, parents);
	EXPECT_FLOAT_EQ(v1.data, 1.0f);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);
	EXPECT_TRUE(v1.getPrev().empty());

	// Test with one previous value
	parents = {prev1};
	Value v2(3.0f, parents);
	EXPECT_FLOAT_EQ(v2.data, 3.0f);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);
	ASSERT_EQ(v2.getPrev().size(), 1);
	EXPECT_EQ(v2.getPrev()[0], prev1);

	// Test with multiple previous values
	parents = {prev1, prev2, prev3};
	Value v3(4.0f, parents);
	EXPECT_FLOAT_EQ(v3.data, 4.0f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);
	ASSERT_EQ(v3.getPrev().size(), 3);
	EXPECT_EQ(v3.getPrev()[0], prev1);
	EXPECT_EQ(v3.getPrev()[1], prev2);
	EXPECT_EQ(v3.getPrev()[2], prev3);

	// Clean up
	delete prev1;
	delete prev2;
	delete prev3;
}

// Test edge cases
TEST(ValueConstructorTest, EdgeCases) {
	// Very large number
	float very_large = 1e30f;
	Value v1(very_large);
	EXPECT_FLOAT_EQ(v1.data, very_large);
	EXPECT_FLOAT_EQ(v1.grad, 0.0f);

	// Very small number
	float very_small = 1e-30f;
	Value v2(very_small);
	EXPECT_FLOAT_EQ(v2.data, very_small);
	EXPECT_FLOAT_EQ(v2.grad, 0.0f);

	// Zero
	Value v3(0.0f);
	EXPECT_FLOAT_EQ(v3.data, 0.0f);
	EXPECT_FLOAT_EQ(v3.grad, 0.0f);

	// Negative zero
	Value v4(-0.0f);
	EXPECT_FLOAT_EQ(v4.data, 0.0f);
	EXPECT_FLOAT_EQ(v4.grad, 0.0f);
}