#include <gtest/gtest.h>
#include "../../src/nn.h"
#include <vector>

/*
	Neuron
*/
// Test Neuron constructor
TEST(NNConstructorTest, NeuronConstructor) {
	Neuron n1(10);
	std::vector<Value*> params = n1.parameters();
	EXPECT_EQ(params.size(), 11);
}

/*
	Layer
*/
// Test Layer constructor
TEST(NNConstructorTest, LayerConstructor) {
	Layer l1(10, 2, "relu");
	std::vector<Value*> params = l1.parameters();
	EXPECT_EQ(params.size(), 22);
}

/*
	MLP
*/
// Test MLP constructor
TEST(NNConstructorTest, MLPConstructor) {
	MLP mlp(10, {2, 1}, "relu");
	std::vector<Value*> params = mlp.parameters();
	EXPECT_EQ(params.size(), 25);
}