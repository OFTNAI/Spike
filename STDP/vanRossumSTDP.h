// vanRossum STDP Class Header
// vanRossumSTDP.h
//
// This STDP learning rule is extracted from the following paper:

//	Rossum, M. C. van, G. Q. Bi, and G. G. Turrigiano. 2000. “Stable Hebbian Learning from Spike Timing-Dependent Plasticity.” The Journal of Neuroscience: The Official Journal of the Society for Neuroscience 20 (23): 8812–21.

// This equation is based upon the multiplicative learning rule without the gaussian random variable
// The default parameters are also those used in the above paper
//	Author: Nasir Ahmad
//	Date: 03/10/2016

#ifndef VANROSSUM_STDP_H
#define VANROSSUM_STDP_H

// Get Synapses & Neurons Class
#include "../Synapses/SpikingSynapses.h"
#include "../Neurons/SpikingNeurons.h"
#include "../STDP/STDP.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>


// STDP Parameters
struct vanrossum_stdp_parameters_struct : stdp_parameters_struct {
	vanrossum_stdp_parameters_struct() : a_minus(0.003), a_plus(7.0f*pow(10.0, -12)), tau_minus(0.02f), tau_plus(0.02f) { } // default Constructor
	// STDP Parameters
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;

};


class vanRossumSTDP : public STDP{
public:

	// Constructor/Destructor
	vanRossumSTDP();
	~vanRossumSTDP();

	struct vanrossum_stdp_parameters_struct* stdp_params;
	SpikingSynapses* syns;
	SpikingNeurons* neurs;
	int* index_of_last_afferent_synapse_to_spike;
	int* d_index_of_last_afferent_synapse_to_spike;
	bool* isindexed_ltd_synapse_spike;
	bool* d_isindexed_ltd_synapse_spike;
	int* index_of_first_synapse_spiked_after_postneuron;
	int* d_index_of_first_synapse_spiked_after_postneuron;

	// Pointers
	virtual void allocate_device_pointers();
	// Pointers
	virtual void reset_STDP_activities();
	// Set STDP Parameters
	virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters);
	// STDP
	virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep);
	// LTP & LTD for this model
	void apply_stdp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);

};


// Kernel to carry out LTP/LTD
__global__ void vanrossum_apply_stdp_to_synapse_weights_kernel(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							int* d_index_of_last_afferent_synapse_to_spike,
							bool* d_isindexed_ltd_synapse_spike,
							int* d_index_of_first_synapse_spiked_after_postneuron,
							struct vanrossum_stdp_parameters_struct stdp_vars,
							float currtime,
							size_t total_number_of_post_neurons);

__global__ void vanrossum_get_indices_to_apply_stdp(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							int* d_index_of_last_afferent_synapse_to_spike,
							bool* d_isindexed_ltd_synapse_spike,
							int* d_index_of_first_synapse_spiked_after_postneuron,
							float currtime,
							size_t total_number_of_synapse);

#endif
