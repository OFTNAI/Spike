#include "VoltageSpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

// VoltageSpikingSynapses Destructor
VoltageSpikingSynapses::~VoltageSpikingSynapses() {
}


void VoltageSpikingSynapses::AddGroup(int presynaptic_group_id, 
                                          int postsynaptic_group_id, 
                                          Neurons * neurons,
                                          Neurons * input_neurons,
                                          float timestep,
                                          synapse_parameters_struct * synapse_params) {
	
	
  SpikingSynapses::AddGroup(presynaptic_group_id, 
                            postsynaptic_group_id, 
                            neurons,
                            input_neurons,
                            timestep,
                            synapse_params);

  voltage_spiking_synapse_parameters_struct * voltage_spiking_synapse_group_params = (voltage_spiking_synapse_parameters_struct*)synapse_params;
  
  // Incrementing number of synapses
  VoltageSpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);
}

void VoltageSpikingSynapses::increment_number_of_synapses(int increment) {
}


void VoltageSpikingSynapses::shuffle_synapses() {
  SpikingSynapses::shuffle_synapses();
}

void VoltageSpikingSynapses::state_update(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
  backend()->state_update(neurons, input_neurons, current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(VoltageSpikingSynapses);
