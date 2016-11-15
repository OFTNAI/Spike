// Evans STDP Class Header
// EvansSTDP.h
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#ifndef Evans_STDP_H
#define Evans_STDP_H

// Get Synapses Class
#include "../Synapses/Synapses.h"
#include "../STDP/STDP.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

//CUDA #include <cuda.h>


// STDP Parameters
struct evans_stdp_parameters_struct : stdp_parameters_struct {
	evans_stdp_parameters_struct() : decay_term_tau_D(0.005f), model_parameter_alpha_D(0.5f), synaptic_neurotransmitter_concentration_alpha_C(0.5f), decay_term_tau_C(0.004f), learning_rate_rho(0.1) { } // default Constructor
	// STDP Parameters
	float decay_term_tau_D;
	float model_parameter_alpha_D;
	float synaptic_neurotransmitter_concentration_alpha_C;
	float decay_term_tau_C;
	float learning_rate_rho;

};


class EvansSTDP : public STDP{
public:

	// Constructor/Destructor
	EvansSTDP();
	~EvansSTDP();

	struct evans_stdp_parameters_struct* stdp_params;
	SpikingSynapses* syns;
	SpikingNeurons* neurs;

	//(NEURON-WISE)
	float * recent_postsynaptic_activities_D;
	float * d_recent_postsynaptic_activities_D;

	//(SYNAPSE-WISE)
	float * recent_presynaptic_activities_C;
	float * d_recent_presynaptic_activities_C;

	// Set STDP Parameters
	virtual void Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters);
	// STDP
	virtual void allocate_device_pointers();
	virtual void reset_STDP_activities();
	
	virtual void Run_STDP(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds, float timestep);
	
	// Updates for this model
	void update_presynaptic_activities(float timestep, float current_time_in_seconds);
	void update_synaptic_efficacies_or_weights(float current_time_in_seconds, float * d_last_spike_time_of_each_neuron);
	void update_postsynaptic_activities(float timestep, float current_time_in_seconds);
};

/*CUDA
__global__ void update_postsynaptic_activities_kernel(float timestep,
								size_t total_number_of_neurons,
								float * d_recent_postsynaptic_activities_D,
								float * d_last_spike_time_of_each_neuron,
								float current_time_in_seconds,
								float decay_term_tau_D,
								float model_parameter_alpha_D);

__global__ void update_presynaptic_activities_C_kernel(float* d_recent_presynaptic_activities_C,
							float* d_time_of_last_spike_to_reach_synapse,
							bool* d_stdp,
							float timestep,
							float current_time_in_seconds,
							size_t total_number_of_synapses,
							float synaptic_neurotransmitter_concentration_alpha_C,
							float decay_term_tau_C);

__global__ void update_synaptic_efficacies_or_weights_kernel(float * d_recent_presynaptic_activities_C,
																float * d_recent_postsynaptic_activities_D,
																int* d_postsynaptic_neuron_indices,
																float* d_synaptic_efficacies_or_weights,
																float current_time_in_seconds,
																float * d_time_of_last_spike_to_reach_synapse,
																float * d_last_spike_time_of_each_neuron,
																bool* d_stdp,
																size_t total_number_of_synapses,
																float learning_rate_rho);
*/

#endif
