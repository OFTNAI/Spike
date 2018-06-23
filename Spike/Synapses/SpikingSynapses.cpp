#include "SpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

SpikingSynapses::~SpikingSynapses() {
#ifdef CRAZY_DEBUG
  std::cout << "SpikingSynapses::~SpikingSynapses\n";
#endif
  free(delays);
  free(biological_conductance_scaling_constants_lambda);

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
	
	if (delay_range_in_timesteps[0] > maximum_axonal_delay_in_timesteps){
		maximum_axonal_delay_in_timesteps = delay_range_in_timesteps[0];
	} else if (delay_range_in_timesteps[1] > maximum_axonal_delay_in_timesteps){
		maximum_axonal_delay_in_timesteps = delay_range_in_timesteps[1];
	}
	if (delay_range_in_timesteps[0] < minimum_axonal_delay_in_timesteps){
		minimum_axonal_delay_in_timesteps = delay_range_in_timesteps[0];
	} else if (delay_range_in_timesteps[1] < minimum_axonal_delay_in_timesteps){
		minimum_axonal_delay_in_timesteps = delay_range_in_timesteps[1];
	}


	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++){
		// Setup Delays
		float delayval = delay_range_in_timesteps[0];
		if (delay_range_in_timesteps[0] != delay_range_in_timesteps[1])
			delayval = delay_range_in_timesteps[0] + (delay_range_in_timesteps[1] - delay_range_in_timesteps[0]) * ((float)rand() / (RAND_MAX));
		delays[i] = round(delayval);
    		biological_conductance_scaling_constants_lambda[i] = spiking_synapse_group_params->biological_conductance_scaling_constant_lambda;
		syn_labels[i] = 0; // Conductance or other systems can now use this if they wish
	}
    	if (neurons->total_number_of_neurons > neuron_pop_size)
      		neuron_pop_size = neurons->total_number_of_neurons; 

}

void SpikingSynapses::increment_number_of_synapses(int increment) {
  delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
  biological_conductance_scaling_constants_lambda = (float*)realloc(biological_conductance_scaling_constants_lambda, total_number_of_synapses * sizeof(float));
  syn_labels = (int*)realloc(syn_labels, total_number_of_synapses * sizeof(int));
}


void SpikingSynapses::shuffle_synapses() {
	
	Synapses::shuffle_synapses();

	int * temp_delays = (int *)malloc(total_number_of_synapses*sizeof(int));
  	float * temp_biological_conductance_scaling_constants_lambda = (float *)malloc(total_number_of_synapses*sizeof(float));
  	int * temp_syn_labels = (int *)malloc(total_number_of_synapses*sizeof(int));
	for(int i = 0; i < total_number_of_synapses; i++) {

		temp_delays[i] = delays[original_synapse_indices[i]];
    		temp_biological_conductance_scaling_constants_lambda[i] = biological_conductance_scaling_constants_lambda[original_synapse_indices[i]];
    		temp_syn_labels[i] = syn_labels[original_synapse_indices[i]];

	}

	delays = temp_delays;
  	biological_conductance_scaling_constants_lambda = temp_biological_conductance_scaling_constants_lambda;
	syn_labels = temp_syn_labels;

}


void SpikingSynapses::state_update(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
  backend()->state_update(neurons, input_neurons, current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(SpikingSynapses);
