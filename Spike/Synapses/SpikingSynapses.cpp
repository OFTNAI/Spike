#include "SpikingSynapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

SpikingSynapses::~SpikingSynapses() {
  free(delays);
  free(stdp);
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

	spiking_synapse_parameters_struct * spiking_synapse_group_params = (spiking_synapse_parameters_struct*)synapse_params;

	for (int i = (total_number_of_synapses - temp_number_of_synapses_in_last_group); i < total_number_of_synapses; i++){
		
		// Convert delay range from time to number of timesteps
		int delay_range_in_timesteps[2] = {int(round(spiking_synapse_group_params->delay_range[0]/timestep)), int(round(spiking_synapse_group_params->delay_range[1]/timestep))};

		// Check delay range bounds greater than timestep
		if ((delay_range_in_timesteps[0] < 1) || (delay_range_in_timesteps[1] < 1)) {
			printf("%d\n", delay_range_in_timesteps[0]);
			printf("%d\n", delay_range_in_timesteps[1]);
			print_message_and_exit("Delay range must be at least one timestep.");
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

		//Set STDP on or off for synapse
		stdp[i] = spiking_synapse_group_params->stdp_on;
	}

}

void SpikingSynapses::increment_number_of_synapses(int increment) {

	Synapses::increment_number_of_synapses(increment);

    delays = (int*)realloc(delays, total_number_of_synapses * sizeof(int));
    stdp = (bool*)realloc(stdp, total_number_of_synapses * sizeof(bool));

}


void SpikingSynapses::reset_state() {
  //CUDA CudaSafeCall(cudaMemset(d_spikes_travelling_to_synapse, 0, sizeof(int)*total_number_of_synapses));
  // Set last spike times to -1000 so that the times do not affect current simulation.
  float* last_spike_to_reach_synapse;
  last_spike_to_reach_synapse = (float*)malloc(sizeof(float)*total_number_of_synapses);
  for (int i=0; i < total_number_of_synapses; i++){
    last_spike_to_reach_synapse[i] = -1000.0f;
  }
  //CUDA CudaSafeCall(cudaMemcpy(d_time_of_last_spike_to_reach_synapse, last_spike_to_reach_synapse, total_number_of_synapses*sizeof(float), cudaMemcpyHostToDevice));
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
  /*CUDA
	if (neurons->high_fidelity_spike_flag){
		check_bitarray_for_presynaptic_neuron_spikes<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(
								d_presynaptic_neuron_indices,
								d_delays,
								neurons->d_bitarray_of_neuron_spikes,
								input_neurons->d_bitarray_of_neuron_spikes,
								neurons->bitarray_length,
								neurons->bitarray_maximum_axonal_delay_in_timesteps,
								current_time_in_seconds,
								timestep,
								total_number_of_synapses,
								d_time_of_last_spike_to_reach_synapse);
		CudaCheckError();
	}
	else{
		move_spikes_towards_synapses_kernel<<<number_of_synapse_blocks_per_grid, threads_per_block>>>(d_presynaptic_neuron_indices,
																			d_delays,
																			d_spikes_travelling_to_synapse,
																			neurons->d_last_spike_time_of_each_neuron,
																			input_neurons->d_last_spike_time_of_each_neuron,
																			current_time_in_seconds,
																			total_number_of_synapses,
																			d_time_of_last_spike_to_reach_synapse);
		CudaCheckError();
	}
  */
}



void SpikingSynapses::calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) {

}

void SpikingSynapses::update_synaptic_conductances(float timestep, float current_time_in_seconds) {

}

