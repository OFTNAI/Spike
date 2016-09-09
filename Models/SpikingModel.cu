#include "SpikingModel.h"

#include "../Helpers/TerminalHelpers.h"


// SpikingModel Constructor
SpikingModel::SpikingModel () {

	timestep = 0.0001f;

	spiking_synapses = NULL;
	spiking_neurons = NULL;
	input_spiking_neurons = NULL;
	stdp_rule = NULL;
}


// SpikingModel Destructor
SpikingModel::~SpikingModel () {


}


void SpikingModel::SetTimestep(float timestep_parameter){

	if ((spiking_synapses == NULL) || (spiking_synapses->total_number_of_synapses == 0)) {
		timestep = timestep_parameter;
	} else {
		print_message_and_exit("You must set the timestep before creating any synapses.");
	}
}


int SpikingModel::AddNeuronGroup(neuron_parameters_struct * group_params) {

	if (spiking_neurons == NULL) print_message_and_exit("Please set neurons pointer before adding neuron groups.");

	int neuron_group_id = spiking_neurons->AddGroup(group_params);
	return neuron_group_id;

}


int SpikingModel::AddInputNeuronGroup(neuron_parameters_struct * group_params) {

	if (input_spiking_neurons == NULL) print_message_and_exit("Please set input_neurons pointer before adding inputs groups.");

	int input_group_id = input_spiking_neurons->AddGroup(group_params);
	return input_group_id;

}


void SpikingModel::AddSynapseGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params) {

	if (spiking_synapses == NULL) print_message_and_exit("Please set synapse pointer before adding synapses.");

	spiking_synapses->AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							spiking_neurons,
							input_spiking_neurons,
							timestep,
							synapse_params);
}


void SpikingModel::AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params) {

	for (int i = 0; i < input_spiking_neurons->total_number_of_groups; i++) {

		AddSynapseGroup(CORRECTED_PRESYNAPTIC_ID(i, true), 
							postsynaptic_group_id,
							synapse_params);

	}

}





void SpikingModel::step_1() {

}


void SpikingModel::step_2 (bool is_optimisation) {

}

void SpikingModel::copy_model_to_device(bool high_fidelity_spike_storage) {

		// PUT IN SpikingModel.cu
	TimerWithMessages * timer = new TimerWithMessages("Setting Up Network...\n");

	int threads_per_block_neurons = 512;
	int threads_per_block_synapses = 512;
	spiking_synapses->set_threads_per_block_and_blocks_per_grid(threads_per_block_synapses);
	spiking_neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);
	input_spiking_neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);

	// Provides order of magnitude speedup for LIF (All to all atleast). 
	// Because all synapses contribute to current_injection on every iteration, having all threads in a block accessing only 1 or 2 positions in memory causes massive slowdown.
	// Randomising order of synapses means that each block is accessing a larger number of points in memory.
	// if (temp_model_type == 1) spiking_synapses->shuffle_synapses();

	spiking_synapses->allocate_device_pointers();
	spiking_neurons->allocate_device_pointers(spiking_synapses->maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);
	input_spiking_neurons->allocate_device_pointers(spiking_synapses->maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);
	stdp_rule->allocate_device_pointers();

	spiking_synapses->copy_constants_and_initial_efficacies_to_device();
	spiking_neurons->copy_constants_to_device();
	input_spiking_neurons->copy_constants_to_device();

	timer->stop_timer_and_log_time_and_message("Network Setup.", true);


}



