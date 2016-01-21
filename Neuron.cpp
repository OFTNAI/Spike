//	Neuron Class C++
//	Neuron.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#include "Neuron.h"
#include <stdlib.h>
#include <stdio.h>

// Neuron Constructor
Neuron::Neuron() {
	// Set up the Neuron Population to start at zero
	numNeurons = 0;
	numPopulations = 0;
	// Initialise Pointers
	numperPop = NULL;
	// Izhikevich Parameters for the neurons
	parama = NULL;
	paramb = NULL;
	paramc = NULL;
	paramd = NULL;
	// State Variables for the neurons
	state_v = NULL;
	state_u = NULL;
}

// Neuron Destructor
Neuron::~Neuron() {
	// Just need to free up the memory
	free(numperPop);
	free(parama);
	free(paramb);
	free(paramc);
	free(paramd);
	free(state_v);
	free(state_u);
}

// Add Population Function
//	INPUT:
//		Number of neurons in population
//		Izhikevich parameter list {a, b, c, d}
int Neuron::AddPopulation(int numinpop, float params[4]){
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
	parama = (float*)realloc(parama,(numNeurons*sizeof(float)));
	paramb = (float*)realloc(paramb,(numNeurons*sizeof(float)));
	paramc = (float*)realloc(paramc,(numNeurons*sizeof(float)));
	paramd = (float*)realloc(paramd,(numNeurons*sizeof(float)));
	state_v = (float*)realloc(state_v,(numNeurons*sizeof(float)));
	state_u = (float*)realloc(state_u,(numNeurons*sizeof(float)));

	// Fill the new entries in the pointers
	numperPop[numPopulations-1] = numNeurons;
	for (int i = (numNeurons-numinpop); i < numNeurons; i++){
		// Set the parameters
		parama[i] = params[0];
		paramb[i] = params[1];
		paramc[i] = params[2];
		paramd[i] = params[3];
		// Set state variables
		state_v[i] = -65.0f;
		state_u[i] = 10.0f;
	}
	// Return Population ID
	return (numPopulations-1);
}


