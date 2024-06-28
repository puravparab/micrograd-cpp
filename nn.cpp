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
		w.emplace_back(new Value(distribution(gen)));
	}
	b = new Value(distribution(gen));
};

Neuron::~Neuron() {
	for (Value* weight : w) {
		delete weight;
	}
	delete b;
}

Value Neuron::operator() (const std::vector<Value>& X) const {
	Value act = *b;
	for (size_t i = 0; i < X.size(); i++){
		act = act + (*w[i] * X[i]); // dot product w . x
	}

	// activations
	if (activation == "relu") { return act.relu();} // relu
	if (activation == "tanh") { return act.tanh();} // tanh
	else {return act;} // linear

}

std::vector<Value*> Neuron::parameters () {
	std::vector<Value*> params = w;
	params.push_back(b);
	return params;
}


/*
	Layer
*/

Layer::Layer (int nin, int out, const std::string& activation){
	for (int i = 0; i < out; i++){
		neurons.emplace_back(nin, activation);
	}
}

std::vector<Value> Layer::operator() (const std::vector<Value>& X) {
	std::vector<Value> outs;
	outs.reserve(neurons.size());
	for (const Neuron& n : neurons) {
		outs.push_back(n(X));
	}
	return outs;
}

std::vector<Value*> Layer::parameters() {
	std::vector<Value*> params;
	for (Neuron& neuron : neurons) {
		auto neuron_params = neuron.parameters();
		params.insert(params.end(), neuron_params.begin(), neuron_params.end());
	}
	return params;
}

/*
	MLP
*/

MLP::MLP (int nin, const std::vector<int>& nouts, const std::string& activation){
	std::vector<int> sz = {nin};
	sz.insert(sz.end(), nouts.begin(), nouts.end());
	
	// Create layers with non linear activation function except for the last layer
	for (size_t i = 0; i < nouts.size(); ++i) {
		bool nonlin = i != nouts.size() - 1;
		layers.emplace_back(sz[i], sz[i+1], nonlin? "relu" : "linear");
	}
}

std::vector<Value> MLP::operator() (const std::vector<float>& X) {
	std::vector<Value> input;
	for (float x : X) {
		input.emplace_back(x);
	}
	for (Layer& layer : layers) {
		input = layer(input);
	}
	return input;
}

std::vector<Value*> MLP::parameters() {
	std::vector<Value*> params;
	for (Layer& layer : layers) {
		auto layer_params = layer.parameters();
		params.insert(params.end(), layer_params.begin(), layer_params.end());
	}
	return params;
}