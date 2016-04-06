// ModelNeurons Class Header
// ModelNeurons.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef ModelNeurons_H
#define ModelNeurons_H

#include "Structs.h"

class ModelNeurons{
public:
	// Constructor/Destructor
	ModelNeurons();
	~ModelNeurons();
	// ModelNeuron Params
	int numNeurons;
	int numPopulations;
	int *numperPop;
	int **neuronpop_shapes;
	// Izhikevich Parameters for the neurons
	struct neuron_struct *neuronpop_variables;
	// Functions
	int AddPopulation(int numinpop, struct neuron_struct params, int shape[2]);

};
#endif