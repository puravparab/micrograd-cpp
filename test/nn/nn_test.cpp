#include <gtest/gtest.h>
#include "../../src/nn.h"
#include <vector>

/*
	Neuron
*/
// Test Neuron constructor
TEST(NNConstructorTest, NeuronConstructor) {
	Neuron n1(10);
	Neuron n2(4, "tanh");
	Neuron n3(7, "sigmoid", 12345);
	EXPECT_EQ(n1.parameters().size(), 11);
	EXPECT_EQ(n2.parameters().size(), 5);
	EXPECT_EQ(n3.parameters().size(), 8);
}

/*
	Layer
*/
// Test Layer constructor
TEST(NNConstructorTest, LayerConstructor) {
	Layer l1(10, 2, "relu");
	EXPECT_EQ(l1.parameters().size(), 22);
}

/*
	MLP
*/
// Test MLP constructor
TEST(NNConstructorTest, MLPConstructor) {
	MLP mlp(10, {2, 1}, "relu");
	EXPECT_EQ(mlp.parameters().size(), 25);
}