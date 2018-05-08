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

  current_spiking_synapse_parameters_struct * current_spiking_synapse_group_params = (current_spiking_synapse_parameters_struct*)synapse_params;
  
  if (decay_terms_tau.size() == 0){
  // If a group has not yet been initialized, make it of this type
	decay_terms_tau.push_back(current_spiking_synapse_group_params->decay_term_tau);
	//num_syn_labels++;
  } else {
  // Check if this pair exists, if yes set the syn_labels or create a new syn_label
	bool isfound = false;
	int indextoset = 0;
  	for (int index = 0; index < decay_terms_tau.size(); index++){
		if (decay_terms_tau[index] == current_spiking_synapse_group_params->decay_term_tau){
			isfound = true;
			indextoset = index;
			break;
		}
	}
	if (!isfound){
		decay_terms_tau.push_back(current_spiking_synapse_group_params->decay_term_tau);
		indextoset = num_syn_labels;
		num_syn_labels++;

	}
	// Now set the synapse labels
  	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++) {
  		syn_labels[i] = indextoset;
  	}
  }
}


void CurrentSpikingSynapses::state_update(SpikingNeurons * input_neurons, SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {
  backend()->state_update(input_neurons, neurons, current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(CurrentSpikingSynapses);

