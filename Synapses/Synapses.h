// Synapse Class Header
// Synapse.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef SYNAPSES_H
#define SYNAPSES_H

#include "../Neurons/Neurons.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>



enum CONNECTIVITY_TYPE
{
    CONNECTIVITY_TYPE_ALL_TO_ALL,
    CONNECTIVITY_TYPE_ONE_TO_ONE,
    CONNECTIVITY_TYPE_RANDOM,
    CONNECTIVITY_TYPE_GAUSSIAN,
    CONNECTIVITY_TYPE_IRINA_GAUSSIAN,
    CONNECTIVITY_TYPE_SINGLE
};


// STDP Parameters
// Temporarily Synapse member (should move to SpikingNeurons)
struct stdp_struct {
	stdp_struct(): w_max(60.0f), a_minus(-0.015f), a_plus(0.005f), tau_minus(0.025f), tau_plus(0.015) { } // default Constructor
	// STDP Parameters
	float w_max;
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;
};


class Synapses {

public:

	// Constructor/Destructor
	Synapses();
	~Synapses();

	int total_number_of_synapses;

	// STDP
	// Temporarily Synapse members (should move to SpikingNeurons)
	struct stdp_struct stdp_vars;
	void SetSTDP(float w_max_new,
				float a_minus_new,
				float a_plus_new,
				float tau_minus_new,
				float tau_plus_new);

	// Full Matrices
	int* presynaptic_neuron_indices; // Previously presyns
	int* postsynaptic_neuron_indices; // Previously postsyns
	float* weights;

	// Device pointers
	int* d_presynaptic_neuron_indices;
	int* d_postsynaptic_neuron_indices;
	float* d_weights;

	int temp_number_of_synapses_in_last_group;

	// Synapse Functions
	virtual void AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						int connectivity_type,
						float weight_range[2],
						int delay_range[2],
						bool stdp_on,
						float parameter,
						float parameter_two);

	virtual void initialise_device_pointers();
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void increment_number_of_synapses(int increment);

	dim3 number_of_synapse_blocks_per_grid;
	dim3 threads_per_block;
};
// GAUSS random number generator
double randn (double mu, double sigma);
#endif