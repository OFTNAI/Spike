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

//temp for test_array test
#include "Connections.h"

class Neurons{
public:
	// Constructor/Destructor
	Neurons();
	~Neurons();

	int* d_test_array;

	// Totals
	int total_number_of_neurons;
	int total_number_of_groups;

	// Group parameters, shapes and indices
	neuron_struct *neuron_variables; //Currently actually neuron wise. Should eventually change
	int **group_shapes;
	int *last_neuron_indices_for_each_group;


	// Device Pointers
	neuron_struct* d_neuron_variables;
	float* d_lastspiketime;

	dim3 number_of_neuron_blocks_per_grid;
	dim3 threads_per_block;

	
	// Functions
	int AddGroup(neuron_struct params, int shape[2]);

	void initialise_device_pointers();
	void reset_neuron_variables_and_spikes();
	void set_threads_per_block_and_blocks_per_grid(int threads);

	void poisupdate_wrapper(float* d_randoms, float timestep);

	void genupdate_wrapper(int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries,
							int genblocknum, 
							dim3 threadsPerBlock);

	void spikingneurons_wrapper(float currtime);

	void stateupdate_wrapper(float* current_injection,
							float timestep);



};



#endif