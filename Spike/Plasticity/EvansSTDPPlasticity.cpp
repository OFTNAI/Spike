//	Evans STDP Class C++
//	EvansSTDPPlasticity.cu
//
//	Author: Nasir Ahmad
//	Date: 23/06/2016

#include "EvansSTDPPlasticity.hpp"
#include "../Helpers/TerminalHelpers.hpp"

EvansSTDPPlasticity::EvansSTDPPlasticity(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_plasticity_parameters_struct* stdp_parameters){
  stdp_params = (evans_stdp_plasticity_parameters_struct *)stdp_parameters;
  syns = synapses;
  neurs = neurons;
}

EvansSTDPPlasticity::~EvansSTDPPlasticity() {
  free(recent_postsynaptic_activities_D);
  free(recent_presynaptic_activities_C);
}

void EvansSTDPPlasticity::prepare_backend_late() {
  // Create extra LIF arrays
  recent_postsynaptic_activities_D = (float*)realloc(recent_postsynaptic_activities_D, total_number_of_plastic_synapses);
  recent_presynaptic_activities_C = (float*)realloc(recent_presynaptic_activities_C, total_number_of_plastic_synapses);
  for (int i = 0; i < total_number_of_plastic_synapses; i++){
    recent_postsynaptic_activities_D[i] = 0.0f;
    recent_presynaptic_activities_C[i] = 0.0f;
  }
}

// Run the STDP
void EvansSTDPPlasticity::state_update (float current_time_in_seconds, float timestep){
  // Update
  update_synaptic_efficacies_or_weights(current_time_in_seconds, timestep);
}

void EvansSTDPPlasticity::update_synaptic_efficacies_or_weights(float current_time_in_seconds, float timestep) {
  backend()->update_synaptic_efficacies_or_weights(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(EvansSTDPPlasticity);
