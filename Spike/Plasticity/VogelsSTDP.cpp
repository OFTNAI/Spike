//  Vogels STDP Class C++
//  VogelsSTDP.cu
//


#include "VogelsSTDP.hpp"
#include "../Helpers/TerminalHelpers.hpp"

VogelsSTDP::~VogelsSTDP() {
  free(vogels_memory_trace);
}

void VogelsSTDP::prepare_backend_late() {
  // Allocate memory for the vogels trace
  vogels_memory_trace = (float*)malloc(sizeof(float)*neurs->total_number_of_neurons);

  // Initialize trace
  for (int i=0; i < neurs->total_number_of_neurons; i++){
    vogels_memory_trace[i] = 0.0f;
  }
}

// Implementation of the STDP Rule for Irina's Model
void VogelsSTDP::Set_STDP_Parameters(SpikingSynapses* synapses, SpikingNeurons* neurons, SpikingNeurons* input_neurons, stdp_parameters_struct* stdp_parameters){
	stdp_params = (vogels_stdp_parameters_struct *)stdp_parameters;
	syns = synapses;
	neurs = neurons;
}

// Run the STDP
void VogelsSTDP::Run_STDP(float current_time_in_seconds, float timestep){
  apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

void VogelsSTDP::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
  backend()->apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(VogelsSTDP);
