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
  // TODO: Check that this doesn't need to be called with input neurons ..
  apply_ltd_to_synapse_weights(current_time_in_seconds);
  apply_ltp_to_synapse_weights(current_time_in_seconds);
}

void HigginsSTDP::apply_ltd_to_synapse_weights(float current_time_in_seconds) {
  backend()->apply_ltd_to_synapse_weights(current_time_in_seconds);
}


void HigginsSTDP::apply_ltp_to_synapse_weights(float current_time_in_seconds) {
  backend()->apply_ltp_to_synapse_weights(current_time_in_seconds);
}

MAKE_INIT_BACKEND(HigginsSTDP);
