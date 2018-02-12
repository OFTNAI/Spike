//  vanRossum STDP Class C++
//  vanRossumSTDP.cu
//
//  Author: Nasir Ahmad
//  Date: 03/10/2016


#include "vanRossumSTDPPlasticity.hpp"
#include "../Helpers/TerminalHelpers.hpp"

vanRossumSTDPPlasticity::vanRossumSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters){
	stdp_params = (vanrossum_stdp_plasticity_parameters_struct *)stdp_parameters;
	syns = synapses;
	neurs = neurons;
}

vanRossumSTDPPlasticity::~vanRossumSTDPPlasticity() {
  free(index_of_last_afferent_synapse_to_spike);
  free(isindexed_ltd_synapse_spike);
  free(index_of_first_synapse_spiked_after_postneuron);
}

void vanRossumSTDPPlasticity::prepare_backend_late() {
}


// Run the STDP
void vanRossumSTDPPlasticity::state_update(float current_time_in_seconds, float timestep){
  apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

void vanRossumSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
  backend()->apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(vanRossumSTDPPlasticity
);
