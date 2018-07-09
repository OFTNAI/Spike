//  CustomSTDPPlasticity STDP Class C++
//  CustomSTDPPlasticity.cu
//
//  Author: Nasir Ahmad
//  Date: 03/10/2016


#include "CustomSTDPPlasticity.hpp"
#include "../Helpers/TerminalHelpers.hpp"

CustomSTDPPlasticity::CustomSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters){
  stdp_params = (custom_stdp_plasticity_parameters_struct *)stdp_parameters;
  syns = synapses;
  neurs = neurons;
}

CustomSTDPPlasticity::~CustomSTDPPlasticity() {
}

void CustomSTDPPlasticity::prepare_backend_late() {
}


// Run the STDP
void CustomSTDPPlasticity::state_update(float current_time_in_seconds, float timestep){
  apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

void CustomSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
  backend()->apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(CustomSTDPPlasticity);
