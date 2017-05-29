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
  free(vogels_memory_trace);
}

void VogelsSTDPPlasticity::prepare_backend_late() {
  // Allocate memory for the vogels trace
  vogels_memory_trace = (float*)malloc(sizeof(float)*neurs->total_number_of_neurons);

  // Initialize trace
  for (int i=0; i < neurs->total_number_of_neurons; i++){
    vogels_memory_trace[i] = 0.0f;
  }
}

// Run the STDP
void VogelsSTDPPlasticity::Run_Plasticity(float current_time_in_seconds, float timestep){
  apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

void VogelsSTDPPlasticity::apply_stdp_to_synapse_weights(float current_time_in_seconds, float timestep) {
  backend()->apply_stdp_to_synapse_weights(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(VogelsSTDPPlasticity);
