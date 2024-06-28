#pragma once
#include <iostream>
#include <vector>
#include <functional>
#include <set>

// node in the computational graph
class Value {
	public:
		float data; // data stored in the node
		float grad; // derivative of the output wrt to the value at this node

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
		Value tanh(); // Tanh
		Value relu(); // ReLU
		Value sigmoid(); // Sigmoid

		void backward(); // run backward propogation from this node

	private:
		std::function<void()> _backward;
		std::vector<Value*> _prev; // vector of the nodes that created this node
		void toposort(std::vector<Value*> &topo, std::set<Value*> &visited); //topological sort
};

// Addition
// float + Value
Value operator+ (const float lhs, Value& rhs);

// Multiplication
// float * Value
Value operator* (const float lhs, Value& rhs);