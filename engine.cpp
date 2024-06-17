#include "engine.h"

/* 
	Constructor
*/
Value::Value(float data)
	: data(data), grad(0.0f), _backward([](){}), _prev({}) {};
Value::Value(float data, const std::vector<Value*>& _prev)
	: data(data), grad(0.0f), _backward([](){}), _prev(_prev) {};

/* 
	Addition
*/
// Value + Value
Value Value::operator+ (const Value& rhs){
	Value out = Value(data + rhs.data, std::vector<Value*> {this, (Value*)&rhs});
	out._backward = [this, &rhs, &out]{
		this->grad += out.grad;
		const_cast<Value&>(rhs).grad += out.grad;
	};
	return out;
};
// Value + float
Value Value::operator+ (const float rhs){
	return (*this) + Value(rhs);
};
// float + Value
Value operator+(const float lhs, Value& rhs) {
  return Value(lhs) + rhs;
};

/* 
	Multiplication
*/
// Value * Value
Value Value::operator* (const Value& rhs){
	Value out = Value(data * rhs.data, std::vector<Value*> {this, (Value*)&rhs});
	out._backward = [this, &rhs, &out]{
		this->grad += rhs.data * out.grad;
		const_cast<Value&>(rhs).grad += this->data * out.grad;
	};
	return out;
};
// Value * float
Value Value::operator* (const float rhs){
	return *this * Value(rhs);
};
// float * Value
Value operator* (const float lhs, Value& rhs) {
  return Value(lhs) * rhs;
};

/* 
	Exponentiation
*/
// power (x ^ n)
Value Value::pow (const Value& other){
	Value out = Value(std::pow(data, other.data), std::vector<Value*> {this, (Value*)&other});
	out._backward = [this, &other, &out]{
		this->grad += (other.data * std::pow(data, other.data - 1)) * out.grad;
	};
	return out;
};
// exp (e ^ n)
Value Value::exp (){
	Value out = Value(std::exp(data), {this});
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
	float tanh = (std::exp(2*data) - 1) / (std::exp(2*data) + 1);
	std::cout << tanh << std::endl;
	Value out = Value(tanh, {this});
	out._backward = [this, &out]{
		std::cout << out.data << std::endl;
		this->grad += (1 - std::pow(out.data, 2)) * out.grad;
	};
	return out;
};
// ReLU
Value Value::relu (){
	float relu = (data > 0) ? data : 0;
	Value out = Value(relu, {this});
	out._backward = [this, &out]{
		this->grad += (out.data > 0) ? out.grad : 0;
	};
	return out;
};