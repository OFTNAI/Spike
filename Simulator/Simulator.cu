// 	Simulator Class
// 	Simulator.cu

//	Authors: Nasir Ahmad (7/12/2015), James Isbister (23/3/2016)

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm> // For random shuffle
#include <time.h>

#include "Simulator.h"
#include "../Neurons/InputSpikingNeurons.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/TimerWithMessages.h"



// Constructor
Simulator::Simulator(){
	// Spike Generators

	synapses = NULL;
	neurons = NULL;
	input_neurons = NULL;

	number_of_stimuli = 0;
	numEntries = NULL;
	genids = NULL;
	gentimes = NULL;
	// Default parameters
	timestep = 0.001f;

	recording_electrodes = NULL;
	input_recording_electrodes = NULL;

	// Default low fidelity spike storage
	high_fidelity_spike_storage = false;
	
	#ifndef QUIETSTART
		print_line_of_dashes_with_blank_lines_either_side();
		printf("Welcome to the SPIKE.\n");
		print_line_of_dashes_with_blank_lines_either_side();
		fflush(stdout);
	#endif
}


// Destructor
Simulator::~Simulator(){

	free(neurons);
	free(input_neurons);
	free(synapses);

	free(numEntries);
	free(genids);
	free(gentimes);
}



void Simulator::SetTimestep(float timest){

	if ((synapses == NULL) || (synapses->total_number_of_synapses == 0)) {
		timestep = timest;
	} else {
		print_message_and_exit("You must set the timestep before creating any synapses.");
	}
}

void Simulator::SetNeuronType(SpikingNeurons * neurons_parameter) {

	neurons = neurons_parameter;

}

void Simulator::SetInputNeuronType(InputSpikingNeurons * inputs_parameter) {

	input_neurons = inputs_parameter;

}

void Simulator::SetSynapseType(SpikingSynapses * synapses_parameter) {

	synapses = synapses_parameter;

}

void Simulator::SetSTDPType(STDP* stdp_parameter) {
	stdp_rule = stdp_parameter;
}


int Simulator::AddNeuronGroup(neuron_parameters_struct * group_params) {

	if (neurons == NULL) print_message_and_exit("Please call SetNeuronType before adding neuron groups.");

	int neuron_group_id = neurons->AddGroup(group_params);
	return neuron_group_id;

}


int Simulator::AddInputNeuronGroup(neuron_parameters_struct * group_params) {

	if (input_neurons == NULL) print_message_and_exit("Please call SetInputNeuronType before adding inputs groups.");

	int input_group_id = input_neurons->AddGroup(group_params);
	return input_group_id;

}


void Simulator::AddSynapseGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params) {

	if (synapses == NULL) print_message_and_exit("Please call SetSynapseType before adding synapses.");

	synapses->AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							timestep,
							synapse_params);
}

void Simulator::AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, 
							synapse_parameters_struct * synapse_params) {

	for (int i = 0; i < input_neurons->total_number_of_groups; i++) {

		AddSynapseGroup(CORRECTED_PRESYNAPTIC_ID(i, true), 
							postsynaptic_group_id,
							synapse_params);

	}

}


void Simulator::setup_network() {

	TimerWithMessages * timer = new TimerWithMessages("Setting Up Network...\n");

	int threads_per_block_neurons = 512;
	int threads_per_block_synapses = 512;
	synapses->set_threads_per_block_and_blocks_per_grid(threads_per_block_synapses);
	neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);
	input_neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);

	// Provides order of magnitude speedup for LIF (All to all atleast). 
	// Because all synapses contribute to current_injection on every iteration, having all threads in a block accessing only 1 or 2 positions in memory causes massive slowdown.
	// Randomising order of synapses means that each block is accessing a larger number of points in memory.
	// if (temp_model_type == 1) synapses->shuffle_synapses();

	neurons->allocate_device_pointers(synapses->maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);
	input_neurons->allocate_device_pointers(synapses->maximum_axonal_delay_in_timesteps, high_fidelity_spike_storage);
	synapses->allocate_device_pointers();
	stdp_rule->allocate_device_pointers();

	printf("HELLO\n");
	for (int i=0; i < synapses->total_number_of_synapses; i++){
		if (synapses->delays[i] > synapses->maximum_axonal_delay_in_timesteps){
			printf("%d, %d\n", i, synapses->delays[i]);
		}
	}
	// printf("InputNeuronNum: %d\n", input_neurons->total_number_of_neurons);
	printf("BYE\n");

	timer->stop_timer_and_log_time_and_message("Network Setup.", true);
}

void Simulator::setup_recording_electrodes_for_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param) {

	TimerWithMessages * timer = new TimerWithMessages("Setting up recording electrodes for neurons...\n");

	recording_electrodes = new RecordingElectrodes(neurons, "Neurons", number_of_timesteps_per_device_spike_copy_check_param, device_spike_store_size_multiple_of_total_neurons_param, proportion_of_device_spike_store_full_before_copy_param);
	recording_electrodes->initialise_device_pointers();
	recording_electrodes->initialise_host_pointers();

	timer->stop_timer_and_log_time_and_message("Recording Electrodes Setup For Neurons.", true);
}


void Simulator::setup_recording_electrodes_for_input_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param) {

	TimerWithMessages * timer = new TimerWithMessages("Setting Up recording electrodes for input neurons...\n");

	input_recording_electrodes = new RecordingElectrodes(input_neurons, "Input_Neurons", number_of_timesteps_per_device_spike_copy_check_param, device_spike_store_size_multiple_of_total_neurons_param, proportion_of_device_spike_store_full_before_copy_param);
	input_recording_electrodes->initialise_device_pointers();
	input_recording_electrodes->initialise_host_pointers();

	timer->stop_timer_and_log_time_and_message("Recording Electrodes Setup For Input Neurons.", true);
}


void Simulator::RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool record_spikes, bool save_recorded_spikes_to_file, SpikeAnalyser *spike_analyser) {
	bool number_of_epochs = 1;
	bool apply_stdp_to_relevant_synapses = false;
	bool count_spikes_per_neuron = true;
	bool present_stimuli_in_random_order = false;

	RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, record_spikes, save_recorded_spikes_to_file, apply_stdp_to_relevant_synapses, count_spikes_per_neuron, present_stimuli_in_random_order, spike_analyser);
}

void Simulator::RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, bool present_stimuli_in_random_order) {

	bool apply_stdp_to_relevant_synapses = true;
	bool count_spikes_per_neuron = false;
	bool record_spikes = false;
	bool save_recorded_spikes_to_file = false;

	RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, record_spikes, save_recorded_spikes_to_file, apply_stdp_to_relevant_synapses, count_spikes_per_neuron, present_stimuli_in_random_order, NULL);
}



void Simulator::RunSimulation(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, bool record_spikes, bool save_recorded_spikes_to_file, bool apply_stdp_to_relevant_synapses, bool count_spikes_per_neuron, bool present_stimuli_in_random_order, SpikeAnalyser *spike_analyser){
	
	int number_of_stimuli = input_neurons->total_number_of_input_stimuli;
	begin_simulation_message(timestep, number_of_stimuli, number_of_epochs, record_spikes, save_recorded_spikes_to_file, present_stimuli_in_random_order, neurons->total_number_of_neurons, input_neurons->total_number_of_neurons, synapses->total_number_of_synapses);
	TimerWithMessages * simulation_timer = new TimerWithMessages();

	if (number_of_epochs == 0) print_message_and_exit("Error. There must be at least one epoch.");

	// SEEDING
	srand(43);

	// STIMULUS ORDER (Put into function + variable)
	int* stimuli_presentation_order;
	stimuli_presentation_order = (int*)malloc(number_of_stimuli*sizeof(int));
	for (int i = 0; i < number_of_stimuli; i++){
		stimuli_presentation_order[i] = i;
	}

	recording_electrodes->write_initial_synaptic_weights_to_file(synapses);
	// recording_electrodes->delete_and_reset_recorded_spikes();

	for (int epoch_number = 0; epoch_number < number_of_epochs; epoch_number++) {
	
		TimerWithMessages * epoch_timer = new TimerWithMessages();
		printf("Starting Epoch: %d\n", epoch_number);

		if (present_stimuli_in_random_order) {
			std::random_shuffle(&stimuli_presentation_order[0], &stimuli_presentation_order[number_of_stimuli]);
		}

		neurons->reset_neurons();
		synapses->reset_synapse_spikes();
		stdp_rule->Reset_STDP();

		float current_time_in_seconds = 0.0f;

		// Running through every Stimulus
		for (int stimulus_index = 0; stimulus_index < number_of_stimuli; stimulus_index++){

			printf("Stimulus: %d, Current time in seconds: %1.2f\n", stimuli_presentation_order[stimulus_index], current_time_in_seconds);

			// These statements have been placed in this order for the Spike Generator Neurons so that they can set up for the next timestep
			input_neurons->current_stimulus_index = stimuli_presentation_order[stimulus_index];
			input_neurons->reset_neurons();

			int number_of_timesteps_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch / timestep;
		
			for (int timestep_index = 0; timestep_index < number_of_timesteps_per_stimulus_per_epoch; timestep_index++){
				
				neurons->reset_current_injections();

				// Carry out the per-timestep computations			
				per_timestep_instructions(current_time_in_seconds, apply_stdp_to_relevant_synapses);

				if (count_spikes_per_neuron) {
					if (recording_electrodes) {
						recording_electrodes->add_spikes_to_per_neuron_spike_count(current_time_in_seconds);
					}
				}

				// // Only save the spikes if necessary
				if (record_spikes){
					if (recording_electrodes) {
						recording_electrodes->collect_spikes_for_timestep(current_time_in_seconds);
						recording_electrodes->copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time_in_seconds, timestep_index, number_of_timesteps_per_stimulus_per_epoch );
					}
					if (input_recording_electrodes) {
						input_recording_electrodes->collect_spikes_for_timestep(current_time_in_seconds);
						input_recording_electrodes->copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time_in_seconds, timestep_index, number_of_timesteps_per_stimulus_per_epoch );
					}
				}

				current_time_in_seconds += float(timestep);

			}

			if (count_spikes_per_neuron) {
				if (spike_analyser) {
					spike_analyser->store_spike_counts_for_stimulus_index(input_neurons->current_stimulus_index, recording_electrodes->d_per_neuron_spike_counts);
				}
			}

			// if (recording_electrodes) printf("Total Number of Spikes: %d\n", recording_electrodes->h_total_number_of_spikes_stored_on_host);

		}
		#ifndef QUIETSTART
		printf("Epoch %d, Complete.\n", epoch_number);
		epoch_timer->stop_timer_and_log_time_and_message(" ", true);
		
		if (record_spikes) {
			if (recording_electrodes) printf(" Number of Spikes: %d\n", recording_electrodes->h_total_number_of_spikes_stored_on_host);
			if (input_recording_electrodes) printf(" Number of Input Spikes: %d\n", input_recording_electrodes->h_total_number_of_spikes_stored_on_host);
		}

		#endif
		// Output Spikes list after each epoch:
		// Only save the spikes if necessary
		if (record_spikes && save_recorded_spikes_to_file){
			printf("Write to file\n");
			if (recording_electrodes) recording_electrodes->write_spikes_to_file(epoch_number);
			if (input_recording_electrodes) input_recording_electrodes->write_spikes_to_file(epoch_number);
		}
	}
	
	// SIMULATION COMPLETE!
	#ifndef QUIETSTART
	simulation_timer->stop_timer_and_log_time_and_message("Simulation Complete!", true);
	#endif

	recording_electrodes->save_network_state(synapses);

	// delete recording_electrodes;
	// delete input_recording_electrodes;

}


void Simulator::per_timestep_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses){

	neurons->check_for_neuron_spikes(current_time_in_seconds, timestep);
	input_neurons->check_for_neuron_spikes(current_time_in_seconds, timestep);

	synapses->interact_spikes_with_synapses(neurons, input_neurons, current_time_in_seconds, timestep);

	synapses->calculate_postsynaptic_current_injection(neurons, current_time_in_seconds, timestep);

	if (apply_stdp_to_relevant_synapses){
		stdp_rule->Run_STDP(neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds, timestep);
	}

	neurons->update_membrane_potentials(timestep);
	input_neurons->update_membrane_potentials(timestep);

}


// // Synapse weight loading
// // INPUT:
// //		Number of weights that you are inputting
// //		The array in which the weights are located
// void Simulator::LoadWeights(int numWeights,
// 						float* newWeights){
// 	// Check if you have the correct number of weights
// 	if (numWeights != synconnects.numsynapses){
// 		print_message_and_exit("The number of weights being loaded is not equivalent to the model.");
// 	}
// 	// Continuing and applying the weights
// 	for (int i=0; i < numWeights; i++){
// 		synconnects.weights[i] = newWeights[i];
// 	}
// }