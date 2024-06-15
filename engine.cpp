#include "engine.h"

Value::Value(float data)
	: data(data), grad(0.0), _backward([](){}), _prev({}) {};
Value::Value(float data, const std::vector<Value*>& _prev)
	: data(data), grad(0.0), _backward([](){}), _prev(_prev) {};

// Add
Value Value::operator+ (const Value& other){
	Value out = Value(data + other.data, std::vector<Value*> {this, (Value*)&other});
	return out;
};