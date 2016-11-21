//	Evans STDP Class C++
//	EvansSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "EvansSTDP.hpp"
#include "../Helpers/TerminalHelpers.hpp"


EvansSTDP::~EvansSTDP() {
  // TODO Check before free
  free(recent_postsynaptic_activities_D);
  free(recent_presynaptic_activities_C);
}

// Implementation of the STDP Rule for Irina's Model
void EvansSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
  stdp_params = (evans_stdp_parameters_struct *)stdp_parameters;
  syns = synapses;
  neurs = neurons;
}

// Run the STDP
void EvansSTDP::Run_STDP(SpikingNeurons* neurons, float current_time_in_seconds, float timestep){
  // TODO: Ensure host and device data are synchronized at this point
  //       OR ***Just pass neurons straight through to update_syn_effs***
  float* d_last_spike_time_of_each_neuron = neurons->last_spike_time_of_each_neuron;
  // Update
  update_synaptic_efficacies_or_weights(current_time_in_seconds, d_last_spike_time_of_each_neuron);
  update_presynaptic_activities(timestep, current_time_in_seconds);
  update_postsynaptic_activities(timestep, current_time_in_seconds);
}

void EvansSTDP::reset_state() {
  backend()->reset_state();
  // STDP::reset_state(); // NB: STDP::reset_state is pure virtual at the moment
}

void EvansSTDP::update_synaptic_efficacies_or_weights(float current_time_in_seconds, float * d_last_spike_time_of_each_neuron) {

  /*CUDA
	update_synaptic_efficacies_or_weights_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
                                                                                                                           d_recent_presynaptic_activities_C,
                                                                                                                           d_recent_postsynaptic_activities_D,
                                                                                                                           syns->d_postsynaptic_neuron_indices,
                                                                                                                           syns->d_synaptic_efficacies_or_weights,
                                                                                                                           current_time_in_seconds,
                                                                                                                           syns->d_time_of_last_spike_to_reach_synapse,
                                                                                                                           d_last_spike_time_of_each_neuron,
                                                                                                                           syns->d_stdp,
                                                                                                                           syns->total_number_of_synapses,
                                                                                                                           stdp_params->learning_rate_rho); // Here learning_rate_rho represents timestep/tau_delta_g in finite difference equation

	CudaCheckError();
  */

}

void EvansSTDP::update_presynaptic_activities(float timestep, float current_time_in_seconds) {

  /*CUDA
	update_presynaptic_activities_C_kernel<<<syns->number_of_synapse_blocks_per_grid, syns->threads_per_block>>>(
							d_recent_presynaptic_activities_C,
							syns->d_time_of_last_spike_to_reach_synapse,
							syns->d_stdp,
							timestep,
							current_time_in_seconds,
							syns->total_number_of_synapses,
							stdp_params->synaptic_neurotransmitter_concentration_alpha_C,
							stdp_params->decay_term_tau_C);

	CudaCheckError();
  */
}

void EvansSTDP::update_postsynaptic_activities(float timestep, float current_time_in_seconds) {

  /*CUDA
	update_postsynaptic_activities_kernel<<<neurs->number_of_neuron_blocks_per_grid, neurs->threads_per_block>>>(
								timestep,
								neurs->total_number_of_neurons,
								d_recent_postsynaptic_activities_D,
								neurs->d_last_spike_time_of_each_neuron,
								current_time_in_seconds,
								stdp_params->decay_term_tau_D,
								stdp_params->model_parameter_alpha_D);

	CudaCheckError();
  */

}

MAKE_PREPARE_BACKEND(EvansSTDP);
