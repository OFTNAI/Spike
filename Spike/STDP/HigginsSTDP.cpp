//	Higgins STDP Class C++
//	HigginsSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "HigginsSTDP.hpp"
#include "../Helpers/TerminalHelpers.hpp"


// STDP Destructor
HigginsSTDP::~HigginsSTDP() {
  free(stdp_params);
  free(syns);
}

// Implementation of the STDP Rule for Irina's Model
void HigginsSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
	stdp_params = (higgins_stdp_parameters_struct *)stdp_parameters;
	syns = synapses;
        neurs = neurons;
}

// Run the STDP
void HigginsSTDP::Run_STDP(float current_time_in_seconds, float timestep){
  // TODO: Ensure host and device data are synchronized at this point
  //       OR ***Just pass neurons straight through to update_syn_effs***
  // float* d_last_spike_time_of_each_neuron = neurons->last_spike_time_of_each_neuron;
  apply_ltd_to_synapse_weights(current_time_in_seconds); // d_last_spike_time_of_each_neuron,
  apply_ltp_to_synapse_weights(current_time_in_seconds); // d_last_spike_time_of_each_neuron, 
}

void HigginsSTDP::apply_ltd_to_synapse_weights(float current_time_in_seconds) { // float* d_last_spike_time_of_each_neuron, 
  backend()->apply_ltd_to_synapse_weights(current_time_in_seconds); // d_last_spike_time_of_each_neuron, 
}


void HigginsSTDP::apply_ltp_to_synapse_weights(float current_time_in_seconds) { // float* d_last_spike_time_of_each_neuron, 
  backend()->apply_ltp_to_synapse_weights(current_time_in_seconds); // d_last_spike_time_of_each_neuron, 
}

MAKE_PREPARE_BACKEND(HigginsSTDP);
