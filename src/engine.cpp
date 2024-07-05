#include "engine.h"

/* 
	Constructor
*/
Value::Value(float data)
	: data(data), grad(0.0f), _backward([](){}), _prev({}) {};
Value::Value(float data, std::vector<Value*>& _prev)
	: data(data), grad(0.0f), _backward([](){}), _prev(_prev) {};


// Backward
void Value::backward(){
	std::vector<Value*> topo;
	std::set<Value*> visited;
	toposort(topo, visited);
	
	grad = 1.0f;
	for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
		Value* node = *it;
		node->_backward();
	}
};

// Topological Sort
void Value::toposort(std::vector<Value*> &topo, std::set<Value*> &visited) {
	if (visited.find(this) != visited.end()){
		return;
	}
	visited.insert(this);
	for (Value* child : _prev){
		child->toposort(topo, visited);
	}
	topo.push_back(this);
};


/* 
	Addition
*/
// Value + Value
Value Value::operator+ (Value& rhs){
	Value* rhs_ptr = &rhs;
	std::vector<Value*> parents = {this, rhs_ptr};
	Value out = Value(data + rhs.data, parents);
	out._backward = [this, rhs_ptr, &out]{
		this->grad += out.grad;
		rhs_ptr->grad += out.grad;
	};
	return out;
};
// Value + float
Value Value::operator+ (float& rhs){
	Value* rhs_ptr = new Value(rhs);
	std::vector<Value*> parents = {this, rhs_ptr};
	Value out = Value(data + rhs_ptr->data, parents);
	out._backward = [this, rhs_ptr, &out]{
		this->grad += out.grad;
		rhs_ptr->grad += out.grad;
	};
	return out;
};
// float + Value
Value operator+(const float& lhs, Value& rhs) {
	Value* lhs_ptr = new Value(lhs);
  return *lhs_ptr + rhs;
};


/* 
	Multiplication
*/
// Value * Value
Value Value::operator* (Value& rhs){
	Value* rhs_ptr = &rhs;
	std::vector<Value*> parents = {this, rhs_ptr};
	Value out = Value(data * rhs.data, parents);
	out._backward = [this, &rhs, &out]{
		this->grad += rhs.data * out.grad;
		rhs.grad += this->data * out.grad;
	};
	return out;
};
// Value * float
Value Value::operator* (const float rhs){
	Value* rhs_ptr = new Value(rhs);
	std::vector<Value*> parents = {this, rhs_ptr};
	Value out = Value(data * rhs_ptr->data, parents);
	out._backward = [this, rhs_ptr, &out]{
		this->grad += rhs_ptr->data * out.grad;
		rhs_ptr->grad += this->data * out.grad;
	};
	return out;
};
// float * Value
Value operator* (const float& lhs, Value& rhs) {
	Value* lhs_ptr = new Value(lhs);
  return *lhs_ptr * rhs;
};


/* 
	Exponentiation
*/
// power (x ^ n)
Value Value::pow (Value& other){
	Value* other_ptr = &other;
	std::vector<Value*> parents = {this, other_ptr};
	Value out = Value(std::pow(data, other.data), parents);
	out._backward = [this, &other, &out]{
		this->grad += (other.data * std::pow(data, other.data - 1.0f)) * out.grad;
	};
	return out;
};
// exp (e ^ n)
Value Value::exp (){
	std::vector<Value*> parents = {this};
	Value out = Value(std::exp(data), parents);
	out._backward = [this, &out]{
		this->grad += out.data * out.grad;
	};
	return out;
};


/* 
	Activation Functions
*/
// Tanh
Value Value::tanh (){
	float tanh = (std::exp(2.0f*data) - 1.0f) / (std::exp(2.0f*data) + 1.0f);
	std::vector<Value*> parents = {this};
	Value out = Value(tanh, parents);
	out._backward = [this, &out]{
		this->grad += (1 - std::pow(out.data, 2.0f)) * out.grad;
	};
	return out;
};
// ReLU
Value Value::relu (){
	float relu = (data > 0) ? data : 0;
	std::vector<Value*> parents = {this};
	Value out = Value(relu, parents);
	out._backward = [this, &out]{
		this->grad += (out.data > 0) ? out.grad : 0;
	};
	return out;
};
// Sigmoid
Value Value::sigmoid() {
	float sigmoid = 1.0f / (1.0f + std::exp(-data));
	std::vector<Value*> parents = {this};
	Value out = Value(sigmoid, parents);
	out._backward = [this, &out] {
		this->grad += out.data * (1.0f - out.data) * out.grad;
	};
	return out;
};