//	Neurons Class Header
//	Neurons.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015
//
//  Adapted from NeuronPopulations by Nasir Ahmad and James Isbister
//	Date: 6/4/2016

#ifndef Neurons_H
#define Neurons_H

#include "Structs.h"

class Neurons{
public:
	// Constructor/Destructor
	Neurons();
	~Neurons();

	// Totals
	int total_number_of_neurons;
	int total_number_of_groups;

	// Group parameters, shapes and indices
	neuron_struct *group_parameters; //Currently actually neuron wise. Should eventually change
	int **group_shapes;
	int *last_neuron_indices_for_each_group;
	
	// Functions
	int AddGroup(neuron_struct params, int shape[2]);



};

__global__ void poisupdate(float* d_randoms, 
							struct neuron_struct* neuronpop_variables,
							float timestep,
							size_t numNeurons);
__global__ void genupdate(struct neuron_struct* neuronpop_variables,
							int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries);
__global__ void stateupdate(struct neuron_struct* neuronpop_variables,
							float* currentinj,
							float timestep,
							size_t numNeurons);
__global__ void spikingneurons(struct neuron_struct* neuronpop_variables,
								float* d_lastspiketime,
								float currtime,
								size_t numNeurons);

#endif