//	Neuron Populations Class C++
//	NeuronPopulations.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#include "ModelNeurons.h"
#include <stdlib.h>
#include <stdio.h>

// ModelNeurons Constructor
ModelNeurons::ModelNeurons() {
	// Set up the Neuron Population to start at zero
	numNeurons = 0;
	numPopulations = 0;
	// Initialise Pointers
	numperPop = NULL;
	neuronpop_shapes = NULL;
	// Parameters for the neurons
	neuronpop_variables = NULL;
}

// ModelNeurons Destructor
ModelNeurons::~ModelNeurons() {
	// Just need to free up the memory
	free(numperPop);
	free(neuronpop_variables);
}

// Add Population Function
//	INPUT:
//		Number of neurons in population
//		Izhikevich parameter list {a, b, c, d}
int ModelNeurons::AddPopulation(int numinpop, struct neuron_struct params, int shape[2]){
	// Check that it is within bounds
	if (numinpop < 0){
		printf("\nError: Population must have at least 1 neuron.\n\n");
		exit(-1);
	}
	// Update the numbers
	numNeurons += numinpop;
	++numPopulations;

	// Allocate space for the new neurons
	numperPop = (int*)realloc(numperPop,(numPopulations*sizeof(int)));
	neuronpop_shapes = (int**)realloc(neuronpop_shapes,(numPopulations*sizeof(int*)));
	neuronpop_shapes[numPopulations-1] = (int*)malloc(2*sizeof(int));
	neuronpop_variables = (struct neuron_struct*)realloc(neuronpop_variables, (numNeurons*sizeof(struct neuron_struct)));

	// Fill the new entries in the pointers
	numperPop[numPopulations-1] = numNeurons;
	neuronpop_shapes[numPopulations-1] = shape;
	for (int i = (numNeurons-numinpop); i < numNeurons; i++){
		// Set the parameters
		neuronpop_variables[i] = params;
	}
	// Return Population ID
	return (numPopulations-1);
}


