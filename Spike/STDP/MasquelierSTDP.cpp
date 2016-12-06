//	Masquelier STDP Class C++
//	MasquelierSTDP.cu
//
//	Author: Nasir Ahmad
//	Date: 03/10/2016

#include "MasquelierSTDP.hpp"
#include "../Helpers/TerminalHelpers.hpp"

void MasquelierSTDP::prepare_backend_extra() {
  assert("This is a test; please remove this line" && false);
  
  // Add the correct space for last synapse
  index_of_last_afferent_synapse_to_spike = (int*)malloc(sizeof(int)*neurs->total_number_of_neurons);
  isindexed_ltd_synapse_spike = (bool*)malloc(sizeof(bool)*neurs->total_number_of_neurons);
  index_of_first_synapse_spiked_after_postneuron = (int*)malloc(sizeof(int)*neurs->total_number_of_neurons);

  // Initialize indices
  for (int i=0; i < neurs->total_number_of_neurons; i++){
    index_of_last_afferent_synapse_to_spike[i] = -1;
    isindexed_ltd_synapse_spike[i] = false;
    index_of_first_synapse_spiked_after_postneuron[i] = -1;
  }
}

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
void MasquelierSTDP::Run_STDP(float current_time_in_seconds, float timestep){
  apply_stdp_to_synapse_weights(current_time_in_seconds);
}

void MasquelierSTDP::apply_stdp_to_synapse_weights(float current_time_in_seconds) {
  // TODO: Check that this doesn't need to be called with input neurons ..
  backend()->apply_stdp_to_synapse_weights(current_time_in_seconds);
}

MAKE_PREPARE_BACKEND(MasquelierSTDP);
