#include "ConductanceSpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

// ConductanceSpikingSynapses Destructor
ConductanceSpikingSynapses::~ConductanceSpikingSynapses() {

#ifdef CRAZY_DEBUG
  std::cout << "@@@@@@@@@@ 0 " << synaptic_conductances_g << " \n";
#endif

  free(synaptic_conductances_g);

#ifdef CRAZY_DEBUG
  std::cout << "@@@@@@@@@@ 1\n";
#endif
}


int ConductanceSpikingSynapses::AddGroup(int presynaptic_group_id, 
                                          int postsynaptic_group_id, 
                                          Neurons * neurons,
                                          Neurons * input_neurons,
                                          float timestep,
                                          synapse_parameters_struct * synapse_params) {
	
	
  int groupID = SpikingSynapses::AddGroup(presynaptic_group_id, 
                            postsynaptic_group_id, 
                            neurons,
                            input_neurons,
                            timestep,
                            synapse_params);

  conductance_spiking_synapse_parameters_struct * conductance_spiking_synapse_group_params = (conductance_spiking_synapse_parameters_struct*)synapse_params;
  
  // Incrementing number of synapses
  ConductanceSpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);

  for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++) {
    synaptic_conductances_g[i] = 0.0f;
    //reversal_potentials_Vhat[i] = conductance_spiking_synapse_group_params->reversal_potential_Vhat;
    //decay_terms_tau_g[i] = conductance_spiking_synapse_group_params->decay_term_tau_g;
  }
  if (reversal_potentials_Vhat.size() == 0){
  // If a group has not yet been initialized, make it of this type
	reversal_potentials_Vhat.push_back(conductance_spiking_synapse_group_params->reversal_potential_Vhat);
	decay_terms_tau_g.push_back(conductance_spiking_synapse_group_params->decay_term_tau_g);
	//num_syn_labels++;
  } else {
  // Check if this pair exists, if yes set the syn_labels or create a new syn_label
	bool isfound = false;
	int indextoset = 0;
  	for (int index = 0; index < reversal_potentials_Vhat.size(); index++){
		if (	(reversal_potentials_Vhat[index] == conductance_spiking_synapse_group_params->reversal_potential_Vhat) &&
			(decay_terms_tau_g[index] == conductance_spiking_synapse_group_params->decay_term_tau_g) ){
			isfound = true;
			indextoset = index;
			break;
		}
	}
	if (!isfound){
		reversal_potentials_Vhat.push_back(conductance_spiking_synapse_group_params->reversal_potential_Vhat);
		decay_terms_tau_g.push_back(conductance_spiking_synapse_group_params->decay_term_tau_g);
		indextoset = num_syn_labels;
		num_syn_labels++;

	}
	// Now set the synapse labels
  	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++) {
  		syn_labels[i] = indextoset;
  	}
  }

  return(groupID);
}

void ConductanceSpikingSynapses::increment_number_of_synapses(int increment) {

  synaptic_conductances_g = (float*)realloc(synaptic_conductances_g, total_number_of_synapses * sizeof(float));
  //reversal_potentials_Vhat = (float*)realloc(reversal_potentials_Vhat, total_number_of_synapses * sizeof(float));
  //decay_terms_tau_g = (float*)realloc(decay_terms_tau_g, total_number_of_synapses * sizeof(float));
}


void ConductanceSpikingSynapses::state_update(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
  backend()->state_update(neurons, input_neurons, current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(ConductanceSpikingSynapses);
