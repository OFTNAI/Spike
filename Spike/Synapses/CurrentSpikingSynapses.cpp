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


void CurrentSpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {
  backend()->calculate_postsynaptic_current_injection(neurons, current_time_in_seconds, timestep);
}

// TODO: Perhaps simplify by having all front-end types derive from one base type
//       with shared members (eg: void* _backend and an empty prepare_backend_extra)
MAKE_PREPARE_BACKEND_EXTRA(CurrentSpikingSynapses);

