#include <gtest/gtest.h>
#include "../../src/nn.h"
#include <vector>

/*
	Neuron
*/
// Test Neuron Forward propagation
TEST(NNForwardPassTest, NeuronForwardPass) {
	/*
		parameters: -0.44502, 0.451169, 0.395825, 0.882431 (b)
	*/
	Neuron n(3, "relu", 12345);
	
	std::vector<std::vector<float>> X = {
		{2.0, 3.0, -1.0},
    {3.0, -1.0, 0.5},
    {0.5, 1.0, 1.0},
    {1.0, 1.0, -1.0}
	};
	std::vector<float> expected_results = {
		0.950073, 0.0, 1.506915, 0.492755
	};
	for (size_t i = 0; i < X.size(); ++i) {
		std::vector<Value> input;
		for (float val : X[i]) {
			input.emplace_back(val);
		}
		Value res = n(input);
		EXPECT_NEAR(res.data, expected_results[i], 1e-5);
	}
}

/*
	Layer
*/
// Test Layer Forward propagation
TEST(NNForwardPassTest, LayerForwardPass) {
	/*
		n1 parameters: -0.44502, 0.451169, 0.395825, 0.882431 (b)
		n2 parameters: -0.44502, 0.451169, 0.395825, 0.882431 (b)
	*/
	Layer l(3,2, "relu");
	
	std::vector<std::vector<float>> X = {
		{2.0, 3.0, -1.0},
    {3.0, -1.0, 0.5},
    {0.5, 1.0, 1.0},
    {1.0, 1.0, -1.0}
	};

	std::vector<float> expected_results = {
		0.950073, 0.0, 1.506915, 0.492755
	};

	for (size_t i = 0; i < X.size(); ++i) {
		std::vector<Value> input;
		for (float val : X[i]) {
			input.emplace_back(val);
		}
		std::vector<Value> res = l(input);
		EXPECT_NEAR(res[0].data, expected_results[i], 1e-5);
		EXPECT_NEAR(res[1].data, expected_results[i], 1e-5);
	}
}

/*
	MLP
*/
// Test MLP Forward propagation
TEST(NNForwardPassTest, MLPForwardPass) {
	/*
		Layer 1:
			n1 parameters: -0.44502, 0.451169, 0.395825, 0.882431 (b)
			n2 parameters: -0.44502, 0.451169, 0.395825, 0.882431 (b)
		Layer 2:
			n1 parameters: -0.44502, 0.451169, 0.395825 (b)
	*/
	MLP mlp(3, {2, 1}, "relu");
	
	std::vector<std::vector<float>> X = {
		{0.0, 0.0, 0.0},
		{1.0, 2.0, 3.0},
		{-1.0, -2.0, -3.0},
		{-1.0, 2.0, 3.0}
	};

	std::vector<float> expected_results = {
		0.4012510, 0.41136490037, 0.395825, 0.41683775633
	};

	for (size_t i = 0; i < X.size(); ++i) {
		std::vector<Value> res = mlp(X[i]);
		EXPECT_NEAR(res[0].data, expected_results[i], 1e-5);
	}
}