//  Vogels STDPPlasticity Class C++
//  VogelsSTDPPlasticity.cu
//


#include "VogelsSTDPPlasticity.hpp"
#include "../Helpers/TerminalHelpers.hpp"

VogelsSTDPPlasticity::VogelsSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters){
	stdp_params = (vogels_stdp_plasticity_parameters_struct *)stdp_parameters;
	syns = synapses;
	neurs = neurons;
}

VogelsSTDPPlasticity::~VogelsSTDPPlasticity() {
}

void VogelsSTDPPlasticity::prepare_backend_late() {
}

// Run the STDP
void VogelsSTDPPlasticity::state_update(float current_time_in_seconds, float timestep){
  apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

void VogelsSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
  backend()->apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(VogelsSTDPPlasticity);
