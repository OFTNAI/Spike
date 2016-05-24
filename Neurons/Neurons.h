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

#include <cuda.h>
#include <stdio.h>


struct neuron_parameters_struct {
	neuron_parameters_struct() { }
};

#define PRESYNAPTIC_IS_INPUT( id ) (id < 0 ? true : false)
#define CORRECTED_PRESYNAPTIC_ID(id, is_input) (is_input ? -1 * (id) - 1 : id) 

class Neurons{
public:
	// Constructor/Destructor
	Neurons();
	~Neurons();

	int total_number_of_neurons;
	int total_number_of_groups;

	float* d_current_injections;

	int **group_shapes;
	int *last_neuron_indices_for_each_group;

	int number_of_neurons_in_new_group;

	dim3 number_of_neuron_blocks_per_grid;
	dim3 threads_per_block;

	void set_threads_per_block_and_blocks_per_grid(int threads);

	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	
	virtual void allocate_device_pointers();
	
	virtual void reset_neurons();
	void reset_current_injections();
};

#endif