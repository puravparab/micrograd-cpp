#pragma once
#include <string>
#include <vector>
#include "engine.h"

/*
	Module
*/
class Module {
	public:
		Module ();
		virtual ~Module() = default;
		void zero_grad(); // zero out the gradients for all Value nodes
		virtual std::vector<Value*> parameters(); // return all trainable parameters
};

/* 
	Neuron
*/
class Neuron : public Module {
	private:
		std::vector<Value*> w; // weights
		Value* b; // bias
		std::string activation; // activation function
	public:
		// Constructors
		Neuron(int nin);
		Neuron(int nin, std::string activation);
		Neuron(int nin, std::string activation, unsigned seed);
		~Neuron() override; // destructor
		Value operator() (std::vector<Value>& X) const;
		std::vector<Value*> parameters() override; // return all trainable parameters
};

/* 
	Layer
*/
class Layer : public Module {
	private:
		std::vector<Neuron*> neurons;
	public:
		Layer (int nin, int out);
		Layer (int nin, int out, const std::string& activation);
		~Layer() override = default;
		std::vector<Value> operator() (std::vector<Value>& X);
		std::vector<Value*> parameters() override; // return all trainable parameters
};

/*
	MLP
*/
class MLP : public Module {
	private:
		std::vector<Layer*> layers;
	public:
		MLP (int nin, const std::vector<int>& nouts);
		MLP (int nin, const std::vector<int>& nouts, const std::string& activation);
		~MLP() override = default;
		std::vector<Value> operator() (std::vector<float>& X);
		std::vector<Value*> parameters() override; // return all trainable parameters
};