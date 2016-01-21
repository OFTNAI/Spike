// Neuron Class Header
// Neuron.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef Neuron_H
#define Neuron_H

class Neuron{
public:
	// Constructor/Destructor
	Neuron();
	~Neuron();
	// Neuron Params
	int numNeurons;
	int numPopulations;
	int *numperPop;
	// Izhikevich Parameters for the neurons
	float *parama;
	float *paramb;
	float *paramc;
	float *paramd;
	// State Variables for the neurons
	float *state_v;
	float *state_u;
	// Functions
	int AddPopulation(int numinpop, float params[4]);

};
#endif