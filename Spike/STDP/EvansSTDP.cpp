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

void EvansSTDP::prepare_backend_late() {
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

// Implementation of the STDP Rule for Irina's Model
void EvansSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
  stdp_params = (evans_stdp_parameters_struct *)stdp_parameters;
  syns = synapses;
  neurs = neurons;
}

// Run the STDP
void EvansSTDP::Run_STDP(float current_time_in_seconds, float timestep){
  // Update
  // TODO: Check that this doesn't need to be called with input neurons ..
  update_synaptic_efficacies_or_weights(current_time_in_seconds);
  update_presynaptic_activities(timestep, current_time_in_seconds);
  update_postsynaptic_activities(timestep, current_time_in_seconds);
}

void EvansSTDP::reset_state() {
  backend()->reset_state();
  // STDP::reset_state(); // NB: STDP::reset_state is pure virtual at the moment
}

void EvansSTDP::update_synaptic_efficacies_or_weights(float current_time_in_seconds) {
  backend()->update_synaptic_efficacies_or_weights(current_time_in_seconds);
}

void EvansSTDP::update_presynaptic_activities(float timestep, float current_time_in_seconds) {
  backend()->update_presynaptic_activities(timestep, current_time_in_seconds);
}

void EvansSTDP::update_postsynaptic_activities(float timestep, float current_time_in_seconds) {
  backend()->update_postsynaptic_activities(timestep, current_time_in_seconds);
}

MAKE_INIT_BACKEND(EvansSTDP);
