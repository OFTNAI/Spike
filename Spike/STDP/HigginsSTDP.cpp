//	Higgins STDP Class C++
//	HigginsSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "HigginsSTDP.hpp"
#include "../Helpers/TerminalHelpers.hpp"


// STDP Constructor
HigginsSTDP::HigginsSTDP() {

	stdp_params = NULL;
	syns = NULL;

}

// STDP Destructor
HigginsSTDP::~HigginsSTDP() {
	// free(stdp_params);
	// free(syns);
}

// Implementation of the STDP Rule for Irina's Model
void HigginsSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
	stdp_params = (higgins_stdp_parameters_struct *)stdp_parameters;
	syns = synapses;
}

// Run the STDP
void HigginsSTDP::Run_STDP(SpikingNeurons* neurons, float current_time_in_seconds, float timestep){
  // TODO: Ensure host and device data are synchronized at this point
  //       OR ***Just pass neurons straight through to update_syn_effs***
  float* d_last_spike_time_of_each_neuron = neurons->last_spike_time_of_each_neuron;
  apply_ltd_to_synapse_weights(d_last_spike_time_of_each_neuron, current_time_in_seconds);
  apply_ltp_to_synapse_weights(d_last_spike_time_of_each_neuron, current_time_in_seconds);
}

void HigginsSTDP::apply_ltd_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
  /*CUDA
	izhikevich_apply_ltd_to_synapse_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	syns->d_stdp,
																	d_last_spike_time_of_each_neuron,
																	syns->d_postsynaptic_neuron_indices,
																	current_time_in_seconds,
																	*stdp_params, // Should make device copy?
																	syns->total_number_of_synapses);

	CudaCheckError();
  */
}


void HigginsSTDP::apply_ltp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
  /*CUDA
	izhikevich_apply_ltp_to_synapse_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	syns->d_stdp,
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	*stdp_params, 
																	current_time_in_seconds,
																	syns->total_number_of_synapses);

	CudaCheckError();
  */
}

