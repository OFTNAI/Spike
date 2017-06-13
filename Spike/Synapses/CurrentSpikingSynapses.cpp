#include "CurrentSpikingSynapses.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"

void CurrentSpikingSynapses::AddGroup(int presynaptic_group_id, 
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

}


void CurrentSpikingSynapses::state_update(SpikingNeurons * input_neurons, SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {
  backend()->state_update(input_neurons, neurons, current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(CurrentSpikingSynapses);

