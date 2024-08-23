#include <random>
#include "nn.h"

/*
	Module
*/
Module::Module () {};

void Module::zero_grad (){
	std::vector<Value*> params = parameters();
	for (Value* param : params) {
		if (param) {
			param->grad = 0.0f;
		}
	}
};

std::vector<Value*> Module::parameters () {
	return std::vector<Value*> {};
}

/*
	Neuron
*/
Neuron::Neuron (int nin) : Neuron(nin, "relu", 12345){};
Neuron::Neuron (int nin, std::string activation) : Neuron(nin, activation, 12345){};
Neuron::Neuron (int nin, std::string activation, unsigned seed) : activation(activation){
	// Assign random values for weights initially
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

Value Neuron::operator() (std::vector<Value>& X) const {
	Value act = *b;
	for (size_t i = 0; i < X.size(); i++){
		act = (*w[i] * X[i]) + act; // dot product w . x
	}

	// activations
	if (activation == "relu") { return act.relu();} // relu
	if (activation == "tanh") { return act.tanh();} // tanh
	if (activation == "sigmoid") { return act.sigmoid();}
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
Layer::Layer (int nin, int out) : Layer(nin, out, "relu"){}
Layer::Layer (int nin, int out, const std::string& activation){
	for (int i = 0; i < out; i++){
		Neuron* n = new Neuron(nin, activation);
		neurons.push_back(n);
	}
}

std::vector<Value> Layer::operator() (std::vector<Value>& X) {
	std::vector<Value> outs;
	outs.reserve(neurons.size());
	for (Neuron* n : neurons) {
		outs.push_back(n->operator()(X));
	}
	return outs;
}

std::vector<Value*> Layer::parameters() {
	std::vector<Value*> params;
	for (Neuron* n : neurons) {
		auto neuron_params = n->parameters();
		params.insert(params.end(), neuron_params.begin(), neuron_params.end());
	}
	return params;
}

/*
	MLP
*/
MLP::MLP (int nin, const std::vector<int>& nouts) : MLP(nin, nouts, "relu"){}
MLP::MLP (int nin, const std::vector<int>& nouts, const std::string& activation){
	std::vector<int> sz = {nin};
	sz.insert(sz.end(), nouts.begin(), nouts.end());
	
	// Create layers with non linear activation function except for the last layer
	for (size_t i = 0; i < nouts.size(); ++i) {
		bool nonlin = i != nouts.size() - 1;
		Layer* l = new Layer(sz[i], sz[i+1], nonlin? activation : "linear");
		layers.push_back(l);
	}
}

std::vector<Value> MLP::operator() (std::vector<float>& X) {
	std::vector<Value> input;
	for (float x : X) {
		input.emplace_back(x); // Convert float data to Value instances
	}
	for (Layer* layer : layers) {
		input = layer->operator()(input);
	}
	return input;
}

std::vector<Value*> MLP::parameters() {
	std::vector<Value*> params;
	for (Layer* layer : layers) {
		auto layer_params = layer->parameters();
		params.insert(params.end(), layer_params.begin(), layer_params.end());
	}
	return params;
}