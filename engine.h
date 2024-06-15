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
		
		Value operator+ (const Value& other);

	private:
		std::function<void()> _backward; 
		std::vector<Value*> _prev; // vector of the nodes that created this node
};

