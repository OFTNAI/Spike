#include "SpikingModel.h"

#include "../Helpers/TerminalHelpers.h"


// SpikingModel Constructor
SpikingModel::SpikingModel () {

	timestep = 0.0001f;
	high_fidelity_spike_storage = false;

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



void SpikingModel::finalise_model() {

}


void SpikingModel::create_parameter_arrays() {

}


void SpikingModel::copy_model_to_device() {

	#ifndef SILENCE_MODEL_SETUP
	TimerWithMessages * timer = new TimerWithMessages("Setting Up Network...\n");
	#endif

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

	#ifndef SILENCE_MODEL_SETUP
	timer->stop_timer_and_log_time_and_message("Network Setup.", true);
	#endif


}


void SpikingModel::reset_model_activities() {

	spiking_neurons->reset_neuron_activities();
	input_spiking_neurons->reset_neuron_activities();
	spiking_synapses->reset_synapse_activities();
	stdp_rule->reset_STDP_activities();

}


void SpikingModel::perform_per_timestep_model_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses){

	spiking_neurons->update_membrane_potentials(timestep, current_time_in_seconds);
	input_spiking_neurons->update_membrane_potentials(timestep, current_time_in_seconds);

	spiking_neurons->check_for_neuron_spikes(current_time_in_seconds, timestep);
	input_spiking_neurons->check_for_neuron_spikes(current_time_in_seconds, timestep);

	spiking_synapses->interact_spikes_with_synapses(spiking_neurons, input_spiking_neurons, current_time_in_seconds, timestep);

	spiking_neurons->reset_current_injections();
	spiking_synapses->calculate_postsynaptic_current_injection(spiking_neurons, current_time_in_seconds, timestep);

	if (apply_stdp_to_relevant_synapses){
		stdp_rule->Run_STDP(spiking_neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds, timestep);
	}

}


// PREVIOUS PER TIMESTEP MODEL INSTRUCTIONS. KEEP FOR NOW FOR COMPARISON, ALTHOUGH NEW ORDERING SHOULD WORK + MORE LOGICAL

// void SpikingModel::perform_per_timestep_model_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses){

// 	spiking_neurons->reset_current_injections();

// 	spiking_neurons->check_for_neuron_spikes(current_time_in_seconds, timestep);
// 	input_spiking_neurons->check_for_neuron_spikes(current_time_in_seconds, timestep);

// 	spiking_synapses->interact_spikes_with_synapses(spiking_neurons, input_spiking_neurons, current_time_in_seconds, timestep);

// 	spiking_synapses->calculate_postsynaptic_current_injection(spiking_neurons, current_time_in_seconds, timestep);

// 	if (apply_stdp_to_relevant_synapses){
// 		stdp_rule->Run_STDP(spiking_neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds, timestep);
// 	}

// 	spiking_neurons->update_membrane_potentials(timestep, current_time_in_seconds);
// 	input_spiking_neurons->update_membrane_potentials(timestep, current_time_in_seconds);

// }

