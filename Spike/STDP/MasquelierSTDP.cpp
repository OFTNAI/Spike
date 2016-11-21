//	Masquelier STDP Class C++
//	MasquelierSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 03/10/2016

#include "MasquelierSTDP.hpp"
#include "../Helpers/TerminalHelpers.hpp"

// Implementation of the STDP Rule for Irina's Model
void MasquelierSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
	stdp_params = (masquelier_stdp_parameters_struct *)stdp_parameters;
	syns = synapses;
	neurs = neurons;
}

void MasquelierSTDP::reset_state(){
  STDP::reset_state();
  backend()->reset_state();
}

// Run the STDP
void MasquelierSTDP::Run_STDP(SpikingNeurons* neurons, float current_time_in_seconds, float timestep){
  // TODO: Ensure host and device data are synchronized at this point
  //       OR ***Just pass neurons straight through to update_syn_effs***
  float* d_last_spike_time_of_each_neuron = neurons->last_spike_time_of_each_neuron;
  apply_stdp_to_synapse_weights(d_last_spike_time_of_each_neuron, current_time_in_seconds);
}


void MasquelierSTDP::apply_stdp_to_synapse_weights(float* d_last_spike_time_of_each_neuron, float current_time_in_seconds) {
  // First reset the indices array
  // In order to carry out nearest spike potentiation only, we must find the spike arriving at each neuron which has the smallest time diff
  /*CUDA
    get_indices_to_apply_stdp<<<neurs->number_of_neuron_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	syns->d_stdp,
																	syns->d_time_of_last_spike_to_reach_synapse,
																	d_index_of_last_afferent_synapse_to_spike,
																	d_isindexed_ltd_synapse_spike,
																	d_index_of_first_synapse_spiked_after_postneuron,
																	current_time_in_seconds,
																	syns->total_number_of_synapses);
	CudaCheckError();

	apply_stdp_to_synapse_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
																	syns->d_postsynaptic_neuron_indices,
																	d_last_spike_time_of_each_neuron,
																	syns->d_stdp,
																	syns->d_time_of_last_spike_to_reach_synapse,
																	syns->d_synaptic_efficacies_or_weights,
																	d_index_of_last_afferent_synapse_to_spike,
																	d_isindexed_ltd_synapse_spike,
																	d_index_of_first_synapse_spiked_after_postneuron,
																	*stdp_params, 
																	current_time_in_seconds,
																	neurs->total_number_of_neurons);
	CudaCheckError();
  */
}

