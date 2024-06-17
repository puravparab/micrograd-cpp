#pragma once
#include <iostream>
#include <vector>
#include <functional>

// node in the computational graph
class Value {
	public:
		float data; // data stored in the node
		float grad; // derivative of the output wrt to the value at this node

		std::function<void()> _backward;

		// Constructor
		Value (float data);
		Value (float data, const std::vector<Value*>& _prev);
		
		// Addition
		Value operator+ (const Value& rhs); // Value + Value
		Value operator+ (const float rhs); // Value + float

		// Multiplication
		Value operator* (const Value& rhs); // Value * float
		Value operator* (const float rhs); // Value * float

		// Exponentiation
		Value pow(const Value& other); // x ^ n
		Value exp(); // e ^ x

		// Activation functions
		Value tanh(); // tanh
		Value relu(); // tanh

	private:
		std::vector<Value*> _prev; // vector of the nodes that created this node
};

// Addition
// float + Value
Value operator+ (const float lhs, Value& rhs);

// Multiplication
// float * Value
Value operator* (const float lhs, Value& rhs);