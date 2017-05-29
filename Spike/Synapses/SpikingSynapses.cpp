#include "SpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

SpikingSynapses::~SpikingSynapses() {
#ifdef CRAZY_DEBUG
  std::cout << "SpikingSynapses::~SpikingSynapses\n";
#endif
  free(delays);
  free(plastic);

  for (int plasticity_id=0; plasticity_id < plasticity_synapse_number_per_rule.size(); plasticity_id++){
  	free(plasticity_synapse_indices_per_rule[plasticity_id]);
  }
}

// Connection Detail implementation
//	INPUT:
//		Pre-neuron population ID
//		Post-neuron population ID
//		An array of the exclusive sum of neuron populations
//		CONNECTIVITY_TYPE (Constants.h)
//		2 number float array for weight range
//		2 number float array for delay range
//		Boolean value to indicate if population is STDP based
//		Parameter = either probability for random synapses or S.D. for Gaussian
void SpikingSynapses::AddGroup(int presynaptic_group_id, 
						int postsynaptic_group_id, 
						Neurons * neurons,
						Neurons * input_neurons,
						float timestep,
						synapse_parameters_struct * synapse_params) {
	
	
	Synapses::AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							timestep,
							synapse_params);

	// First incrementing the synapses
	SpikingSynapses::increment_number_of_synapses(temp_number_of_synapses_in_last_group);

	spiking_synapse_parameters_struct * spiking_synapse_group_params = (spiking_synapse_parameters_struct*)synapse_params;

	// Store STDP Rule as necessary
	int plasticity_id = -1;
	int original_num_plasticity_indices = 0;
	if (spiking_synapse_group_params->plasticity_ptr != nullptr){
		plasticity_id = spiking_synapse_group_params->plasticity_ptr->plasticity_rule_id;
		// Store or recall STDP Pointer
		// Check if this pointer has already been stored
		if (plasticity_id < 0){
			plasticity_id = plasticity_rule_vec.size();
			plasticity_rule_vec.push_back(spiking_synapse_group_params->plasticity_ptr);
			// Allocate space to store stdp indices
			plasticity_synapse_indices_per_rule.push_back(nullptr);
			// plasticity_synapse_indices_per_rule = (int**)realloc(plasticity_synapse_indices_per_rule, plasticity_rule_vec.size() * sizeof(int*));
			// plasticity_synapse_indices_per_rule[plasticity_id] = nullptr;
			plasticity_synapse_number_per_rule.push_back(0);
			// Apply ID to STDP class
			spiking_synapse_group_params->plasticity_ptr->plasticity_rule_id = plasticity_id;
		}

		// Allocate memory for the new incoming synapses
		original_num_plasticity_indices = plasticity_synapse_number_per_rule[plasticity_id];
		plasticity_synapse_number_per_rule[plasticity_id] += temp_number_of_synapses_in_last_group;
		plasticity_synapse_indices_per_rule[plasticity_id] = (int*)realloc(plasticity_synapse_indices_per_rule[plasticity_id], plasticity_synapse_number_per_rule[plasticity_id] * sizeof(int));
	}

	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++){
		
		// Convert delay range from time to number of timesteps
		int delay_range_in_timesteps[2] = {int(round(spiking_synapse_group_params->delay_range[0]/timestep)), int(round(spiking_synapse_group_params->delay_range[1]/timestep))};

		// Check delay range bounds greater than timestep
		if ((delay_range_in_timesteps[0] < 1) || (delay_range_in_timesteps[1] < 1)) {
			printf("%d\n", delay_range_in_timesteps[0]);
			printf("%d\n", delay_range_in_timesteps[1]);
#ifdef CRAZY_DEBUG
                        // spiking_synapse_group_params->delay_range[0] = timestep;
                        // spiking_synapse_group_params->delay_range[1] = timestep;
			printf("################### Delay range must be at least one timestep\n");
#else

                        print_message_and_exit("Delay range must be at least one timestep.");
#endif
		}

		// Setup Delays
		if (delay_range_in_timesteps[0] == delay_range_in_timesteps[1]) {
			delays[i] = delay_range_in_timesteps[0];
		} else {
			float random_delay = delay_range_in_timesteps[0] + (delay_range_in_timesteps[1] - delay_range_in_timesteps[0]) * ((float)rand() / (RAND_MAX));
			delays[i] = round(random_delay);
		}

		if (delay_range_in_timesteps[0] > maximum_axonal_delay_in_timesteps){
			maximum_axonal_delay_in_timesteps = delay_range_in_timesteps[0];
		} else if (delay_range_in_timesteps[1] > maximum_axonal_delay_in_timesteps){
			maximum_axonal_delay_in_timesteps = delay_range_in_timesteps[1];
		}

		//Set STDP on or off for synapse (now using stdp id)
		plastic[i] = false;
		if (plasticity_id >= 0){
			plastic[i] = true;
			plasticity_synapse_indices_per_rule[plasticity_id][original_num_plasticity_indices + (i  - (total_number_of_synapses - temp_number_of_synapses_in_last_group))] = i;
		}
	}

}

void SpikingSynapses::increment_number_of_synapses(int increment) {
  delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
  plastic = (bool*)realloc(plastic, total_number_of_synapses * sizeof(bool));
}


void SpikingSynapses::shuffle_synapses() {
	
	Synapses::shuffle_synapses();

	int * temp_delays = (int *)malloc(total_number_of_synapses*sizeof(int));
	bool * temp_plastic = (bool *)malloc(total_number_of_synapses*sizeof(bool));
	for(int i = 0; i < total_number_of_synapses; i++) {

		temp_delays[i] = delays[original_synapse_indices[i]];
		temp_plastic[i] = plastic[original_synapse_indices[i]];

	}

	delays = temp_delays;
	plastic = temp_plastic;

}


void SpikingSynapses::interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
  backend()->interact_spikes_with_synapses(neurons, input_neurons, current_time_in_seconds, timestep);
}

SPIKE_MAKE_STUB_INIT_BACKEND(SpikingSynapses);
