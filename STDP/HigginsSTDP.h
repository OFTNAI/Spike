// Higgins STDP Class Header
// HigginsSTDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef HIGGINS_STDP_H
#define HIGGINS_STDP_H

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
struct higgins_stdp_parameters_struct : stdp_parameters_struct {
	higgins_stdp_parameters_struct() : w_max(60.0f), a_minus(-0.015f), a_plus(0.005f), tau_minus(0.025f), tau_plus(0.015) { } // default Constructor
	// STDP Parameters
	float w_max;
	float a_minus;
	float a_plus;
	float tau_minus;
	float tau_plus;

};


class HigginsSTDP : public STDP{
public:

	// Constructor/Destructor
	HigginsSTDP();
	~HigginsSTDP();

	struct higgins_stdp_parameters_struct* stdp_params;
	SpikingSynapses* syns;

	// Set STDP Parameters
	virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters);
	// STDP
	virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep);
	// LTP & LTD for this model
	void apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);
	void apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds);

};


// Kernels to carry out LTP/LTD
__global__ void izhikevich_apply_ltd_to_synapse_weights_kernel(float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							bool* d_stdp,
							float* d_last_spike_time_of_each_neuron,
							int* d_postsyns,
							float currtime,
							struct higgins_stdp_parameters_struct stdp_vars,
							size_t total_number_of_synapse);

__global__ void izhikevich_apply_ltp_to_synapse_weights_kernel(int* d_postsyns,
							float* d_last_spike_time_of_each_neuron,
							bool* d_stdp,
							float* d_time_of_last_spike_to_reach_synapse,
							float* d_synaptic_efficacies_or_weights,
							struct higgins_stdp_parameters_struct stdp_vars,
							float currtime,
							size_t total_number_of_synapse);

#endif