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

//	CUDA library
#include <cuda.h>

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

	void poisupdate_wrapper(float* d_randoms, 
							neuron_struct* neuronpop_variables,
							float timestep,
							size_t numNeurons, 
							dim3 vectorblocksPerGrid,
							dim3 threadsPerBlock);

	void genupdate_wrapper(neuron_struct* neuronpop_variables,
							int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries,
							int genblocknum, 
							dim3 threadsPerBlock);

	void spikingneurons_wrapper(neuron_struct* d_neuron_group_parameters,
								float* d_lastspiketime,
								float currtime,
								size_t numNeurons,
								dim3 vectorblocksPerGrid, 
								dim3 threadsPerBlock);

	void stateupdate_wrapper(neuron_struct* d_neuronpop_variables,
							float* current_injection,
							float timestep,
							size_t total_number_of_neurons,
							dim3 vectorblocksPerGrid, 
							dim3 threadsPerBlock);

};



#endif