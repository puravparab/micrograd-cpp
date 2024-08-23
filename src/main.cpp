#include <iostream>
#include <iomanip>
#include <vector>
#include "engine.h"
#include "nn.h"

void printWelcomeMessage() {
	const int width = 50;
	const char fillChar = '-';
	std::cout << std::setfill(fillChar) << std::setw(width) << "" << std::endl;
	std::cout << std::setfill(' ');
	std::cout << std::setw((width - 23) / 2) << "" << "Micrograd in C++" << std::endl;
	std::cout << std::setfill(fillChar) << std::setw(width) << "" << std::endl;
	std::cout << std::setfill(' ');
	std::cout << std::left << std::setw(15) << "  Author:" 
						<< std::right << std::setw(35) << "Purav Parab" << std::endl;
	std::cout << std::setfill(fillChar) << std::setw(width) << "" << std::endl;
	std::cout << std::endl;
}

// int main(){
// 	Value A(2.0f);
// 	Value B(2.0f);

// 	Value C = B.tanh();
// 	Value D = C.pow(A);

// 	D.backward();

// 	std::cout << "A: " << A.data << " -> " << A.grad << std::endl;
// 	std::cout << "B: " << B.data << " -> " << B.grad << std::endl;
// 	std::cout << "C: " << C.data << " -> " << C.grad << std::endl;
// 	std::cout << "D: " << D.data << " -> " << D.grad << std::endl;
// 	return 0;
// };

int main() {
	printWelcomeMessage();

	std::vector<std::vector<float>> X = {
		{2.0, 3.0, -1.0},
    {3.0, -1.0, 0.5},
    {0.5, 1.0, 1.0},
    {1.0, 1.0, -1.0}
	};

	std::vector<float> Y = {1.0, -1.0, -1.0, 1.0};

	// MLP nn(3, {4, 4, 1}, "relu");
	// Neuron n(3, "relu");
	// for (auto& input: n.parameters()){
	// 	std::cout << input->data << std::endl;
	// }
	// std::vector<Value> XV;
	// for (float& val: X[0]) {
	// 	XV.emplace_back(val);
	// 	std::cout << val << std::endl;
	// }
	// Value out = n(XV);
	// std::cout << out.data << ", " << out.grad << std::endl;

	// Neuron n(3, "relu");
	// // // gradient descent
	// for (size_t i = 0; i < 20; i++){
	// 	// forward pass
	// 	std::vector<Value> ypred;
	// 	for (const auto& x : X) {
	// 		std::vector<Value> xv;
	// 		for (float val : x) {
	// 			xv.emplace_back(val);
	// 		}
	// 		ypred.push_back(n(xv));
	// 	}

	// 	// Compute loss (squared error)
	// 	Value loss(0.0f);
	// 	for (size_t i = 0; i < Y.size(); i++) {
	// 		Value diff = Value(-1.0f * Y[i]) +  ypred[i];
	// 		loss = (diff * diff) + loss;
	// 	}

	// 	// Backward pass
	// 	n.zero_grad();
	// 	loss.backward();
	// 	std::cout << loss.data << std::endl;
	// 	for (auto& input: n.parameters()){
	// 		std::cout << input->grad << std::endl;
	// 	}
		

	// 	// Update parameters
	// 	float learning_rate = -0.05f;
	// 	for (Value* p : n.parameters()) {
	// 		p->data += learning_rate * p->grad;
	// 	}

	// 	std::cout << "Iteration " << i << ", Loss: " << loss.data << std::endl;
	// }
};