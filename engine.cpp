#include "engine.h"

Value::Value(float data)
	: data(data), grad(0.0f), _backward([](){}), _prev({}) {};
Value::Value(float data, const std::vector<Value*>& _prev)
	: data(data), grad(0.0f), _backward([](){}), _prev(_prev) {};

// Add
Value Value::operator+ (const Value& rhs){
	Value out = Value(data + rhs.data, std::vector<Value*> {this, (Value*)&rhs});
	out._backward = [this, &rhs, &out]{
		this->grad += out.grad;
		const_cast<Value&>(rhs).grad += out.grad;
	};
	return out;
};
Value Value::operator+ (const float rhs){
	return (*this) + Value(rhs);
};
Value operator+(const float lhs, Value& rhs) {
  return Value(lhs) + rhs;
};

// Multiply
Value Value::operator* (const Value& rhs){
	Value out = Value(data * rhs.data, std::vector<Value*> {this, (Value*)&rhs});
	out._backward = [this, &rhs, &out]{
		this->grad += rhs.data * out.grad;
		const_cast<Value&>(rhs).grad += this->data * out.grad;
	};
	return out;
};
Value Value::operator* (const float rhs){
	return *this * Value(rhs);
};
Value operator* (const float lhs, Value& rhs) {
  return Value(lhs) * rhs;
};