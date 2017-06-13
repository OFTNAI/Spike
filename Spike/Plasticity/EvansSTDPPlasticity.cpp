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
  recent_postsynaptic_activities_D = (float*)realloc(recent_postsynaptic_activities_D, (neurs->total_number_of_neurons*sizeof(float)));
  recent_presynaptic_activities_C = (float*)realloc(recent_presynaptic_activities_C, syns->total_number_of_synapses*sizeof(float));
  for (int i = 0; i < neurs->total_number_of_neurons; i++){
    recent_postsynaptic_activities_D[i] = 0.0f;
  }
  for (int i = 0; i < syns->total_number_of_synapses; i++){
    recent_presynaptic_activities_C[i] = 0.0f;
  }
}

// Run the STDP
void EvansSTDPPlasticity::state_update (float current_time_in_seconds, float timestep){
  // Update
  update_synaptic_efficacies_or_weights(current_time_in_seconds);
  update_presynaptic_activities(timestep, current_time_in_seconds);
  update_postsynaptic_activities(timestep, current_time_in_seconds);
}

void EvansSTDPPlasticity::update_synaptic_efficacies_or_weights(float current_time_in_seconds) {
  backend()->update_synaptic_efficacies_or_weights(current_time_in_seconds);
}

void EvansSTDPPlasticity::update_presynaptic_activities(float timestep, float current_time_in_seconds) {
  backend()->update_presynaptic_activities(timestep, current_time_in_seconds);
}

void EvansSTDPPlasticity::update_postsynaptic_activities(float timestep, float current_time_in_seconds) {
  backend()->update_postsynaptic_activities(timestep, current_time_in_seconds);
}

SPIKE_MAKE_INIT_BACKEND(EvansSTDPPlasticity);
