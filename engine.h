#pragma once
#include <vector>
#include <functional>

// node in the computational graph
class Value {
	public:
		float data; // data stored in the node
		float grad; // derivative of the output wrt to the value at this node

		Value (float data);
		Value (float data, const std::vector<Value*>& _prev);
		
		Value operator+ (const Value& rhs);
		Value operator+ (const float rhs); // Value + float

		Value operator* (const Value& rhs);
		Value operator* (const float rhs); // Value * float

		std::function<void()> _backward;
	private:
		std::vector<Value*> _prev; // vector of the nodes that created this node
};

// float + Value
Value operator+ (const float lhs, Value& rhs);

// float * Value
Value operator* (const float lhs, Value& rhs);