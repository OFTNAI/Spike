#include "SpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

SpikingSynapses::~SpikingSynapses() {
#ifdef CRAZY_DEBUG
  std::cout << "SpikingSynapses::~SpikingSynapses\n";
#endif
  free(delays);
  free(stdp);

  for (int stdp_id=0; stdp_id < stdp_synapse_number_per_rule.size(); stdp_id++){
  	free(stdp_synapse_indices_per_rule[stdp_id]);
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
	int stdp_id = -1;
	int original_num_stdp_indices = 0;
	if (spiking_synapse_group_params->stdp_ptr != nullptr){
		stdp_id = spiking_synapse_group_params->stdp_ptr->stdp_rule_id;
		// Store or recall STDP Pointer
		// Check if this pointer has already been stored
		if (stdp_id < 0){
			stdp_id = stdp_rule_vec.size();
			stdp_rule_vec.push_back(spiking_synapse_group_params->stdp_ptr);
			// Allocate space to store stdp indices
			stdp_synapse_indices_per_rule.push_back(nullptr);
			// stdp_synapse_indices_per_rule = (int**)realloc(stdp_synapse_indices_per_rule, stdp_rule_vec.size() * sizeof(int*));
			// stdp_synapse_indices_per_rule[stdp_id] = nullptr;
			stdp_synapse_number_per_rule.push_back(0);
			// Apply ID to STDP class
			spiking_synapse_group_params->stdp_ptr->stdp_rule_id = stdp_id;
		}

		// Allocate memory for the new incoming synapses
		original_num_stdp_indices = stdp_synapse_number_per_rule[stdp_id];
		stdp_synapse_number_per_rule[stdp_id] += temp_number_of_synapses_in_last_group;
		stdp_synapse_indices_per_rule[stdp_id] = (int*)realloc(stdp_synapse_indices_per_rule[stdp_id], stdp_synapse_number_per_rule[stdp_id] * sizeof(int));
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
		stdp[i] = false;
		if (stdp_id >= 0){
			stdp[i] = true;
			stdp_synapse_indices_per_rule[stdp_id][original_num_stdp_indices + (i  - (total_number_of_synapses - temp_number_of_synapses_in_last_group))] = i;
		}
	}

}

void SpikingSynapses::increment_number_of_synapses(int increment) {
  delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
  stdp = (bool*)realloc(stdp, total_number_of_synapses * sizeof(bool));
}


void SpikingSynapses::shuffle_synapses() {
	
	Synapses::shuffle_synapses();

	int * temp_delays = (int *)malloc(total_number_of_synapses*sizeof(int));
	bool * temp_stdp = (bool *)malloc(total_number_of_synapses*sizeof(bool));
	for(int i = 0; i < total_number_of_synapses; i++) {

		temp_delays[i] = delays[original_synapse_indices[i]];
		temp_stdp[i] = stdp[original_synapse_indices[i]];

	}

	delays = temp_delays;
	stdp = temp_stdp;

}


void SpikingSynapses::interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
  backend()->interact_spikes_with_synapses(neurons, input_neurons, current_time_in_seconds, timestep);
}

SPIKE_MAKE_STUB_INIT_BACKEND(SpikingSynapses);
