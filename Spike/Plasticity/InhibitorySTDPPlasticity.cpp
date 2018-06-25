//  Inhibitory STDPPlasticity Class C++
//  InhibitorySTDPPlasticity.cu
//


#include "InhibitorySTDPPlasticity.hpp"
#include "../Helpers/TerminalHelpers.hpp"

InhibitorySTDPPlasticity::InhibitorySTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters){
	stdp_params = (inhibitory_stdp_plasticity_parameters_struct *)stdp_parameters;
	syns = synapses;
	neurs = neurons;
}

InhibitorySTDPPlasticity::~InhibitorySTDPPlasticity() {
}

void InhibitorySTDPPlasticity::prepare_backend_late() {
}

// Run the STDP
void InhibitorySTDPPlasticity::state_update(float current_time_in_seconds, float timestep){
  apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

void InhibitorySTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
  backend()->apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(InhibitorySTDPPlasticity);
