// Neuron Class Header
// NeuronPopulations.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef NeuronPopulations_H
#define NeuronPopulations_H

class NeuronPopulations{
public:
	// Constructor/Destructor
	NeuronPopulations();
	~NeuronPopulations();
	// Neuron Params
	int numNeurons;
	int numPopulations;
	int *numperPop;
	// Izhikevich Parameters for the neurons
	struct neuron_struct *neuronpop_variables;
	// Functions
	int AddPopulation(int numinpop, struct neuron_struct params);

};
#endif