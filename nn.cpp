#include <random>
#include "nn.h"

/*
	Module
*/
Module::Module () {};

void Module::zero_grad (){
	std::vector<Value*> params = parameters();
	for (Value* param : params) {
		param->grad = 0.0f;
	}
};

std::vector<Value*> Module::parameters () {
	return std::vector<Value*> {};
}

/*
	Neuron
*/
Neuron::Neuron (int nin) : Neuron(nin, "relu"){};
Neuron::Neuron (int nin, std::string activation) : activation(activation) {
	unsigned seed = std::time(0);
	std::default_random_engine gen(seed);
	std::uniform_real_distribution<float> distribution(-1.0, 1.0);

	w.reserve(nin);
	for (int i = 0; i < nin; i++){
		w.push_back(new Value(distribution(gen)));
	}
	b = new Value(distribution(gen));
};

Neuron::~Neuron() {
	for (Value* weight : w) {
		delete weight;
	}
	delete b;
}

std::vector<Value*> Neuron::parameters () {
	std::vector<Value*> params = w;
	params.push_back(b);
	return params;
}

Value Neuron::operator() (const std::vector<float>& X) {
	Value act = *b;
	for (size_t i = 0; i < X.size(); i++){
		act = act + (*w[i] * Value(X[i])); // dot product w . x
	}

	if (activation == "relu") { return act.relu();}
	if (activation == "tanh") { return act.tanh();}
	else {return act;}

}