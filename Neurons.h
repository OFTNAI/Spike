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
#include <stdio.h>

//temp for test_array test
#include "Connections.h"

struct neuron_parameters_struct {
	neuron_parameters_struct() { }

};



struct neuron_struct {
	neuron_struct(): parama(0.0f), paramb(0.0f), paramc(0.0f), paramd(0.0f), state_v(-70.0f), state_u(0.0f), rate(0.0f) { }   // default Constructor
	float parama;
	float paramb;
	float paramc;
	float paramd;
	// State variables
	float state_v;
	float state_u;
	// Rate for poisson
	float rate;
};

class Neurons{
public:
	// Constructor/Destructor
	Neurons();
	~Neurons();

	// Totals
	int total_number_of_neurons;
	int total_number_of_groups;

	float * d_last_spike_time;

	float* d_current_injections;

	// Group parameters, shapes and indices
	neuron_struct *neuron_variables;
	int **group_shapes;
	int *last_neuron_indices_for_each_group;

	int number_of_neurons_in_new_group;

	// Device Pointers
	neuron_struct* d_neuron_variables;
	float* d_lastspiketime;

	dim3 number_of_neuron_blocks_per_grid;
	dim3 threads_per_block;

	


	// Functions
	virtual int AddGroupNew(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void initialise_device_pointersNew();
	virtual void reset_neuron_variables_and_spikesNew();

	void reset_device_current_injections();

	virtual void set_threads_per_block_and_blocks_per_grid(int threads);




};



#endif