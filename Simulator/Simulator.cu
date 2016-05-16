// 	Simulator Class
// 	Simulator.cu

//	Authors: Nasir Ahmad (7/12/2015), James Isbister (23/3/2016)

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm> // For random shuffle
#include <time.h>

#include "Simulator.h"
#include "../Neurons/GeneratorSpikingNeurons.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"


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
	
	#ifndef QUIETSTART
		// Say Hi to the user:
		printf("\nWelcome to the SPIKE.\n");
		print_line_of_dashes_with_blank_lines_either_side();
		printf("Setting up Neurons and Synapses:\n");
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

void Simulator::SetInputNeuronType(PoissonSpikingNeurons * inputs_parameter) {

	input_neurons = inputs_parameter;

}

void Simulator::SetSynapseType(SpikingSynapses * synapses_parameter) {

	synapses = synapses_parameter;

}



int Simulator::AddNeuronGroup(neuron_parameters_struct * group_params, int group_shape[2]) {

	if (neurons == NULL) print_message_and_exit("Please call SetNeuronType before adding neuron groups.");

	int neuron_group_id = neurons->AddGroup(group_params, group_shape);
	return neuron_group_id;

}


int Simulator::AddInputNeuronGroup(neuron_parameters_struct * group_params, int group_shape[2]) {

	if (input_neurons == NULL) print_message_and_exit("Please call SetInputNeuronType before adding inputs groups.");

	int input_group_id = input_neurons->AddGroup(group_params, group_shape);
	return input_group_id;

}


void Simulator::AddSynapseGroup(int presynaptic_group_id, 
							int postsynaptic_group_id, 
							int connectivity_type,
							float weight_range[2], 
							float delay_range[2],
							bool stdp_on,
							connectivity_parameters_struct * connectivity_params,
							float parameter,
							float parameter_two) {

	if (synapses == NULL) print_message_and_exit("Please call SetSynapseType before adding synapses.");

	
	// Convert delay range from time to number of timesteps
	int delay_range_in_timesteps[2] = {int(round(delay_range[0]/timestep)), int(round(delay_range[1]/timestep))};

	if ((delay_range_in_timesteps[0] < 1) || (delay_range_in_timesteps[1] < 1)) {
		print_message_and_exit("Delay range must be at least one timestep.");
	}

	synapses->AddGroup(presynaptic_group_id, 
							postsynaptic_group_id, 
							neurons,
							input_neurons,
							connectivity_type, 
							weight_range,
							delay_range_in_timesteps,
							stdp_on,
							connectivity_params,
							parameter,
							parameter_two);
}

void Simulator::AddSynapseGroupsForNeuronGroupAndEachInputGroup(int postsynaptic_group_id, 
							int connectivity_type,
							float weight_range[2], 
							float delay_range[2],
							bool stdp_on,
							connectivity_parameters_struct * connectivity_params,
							float parameter,
							float parameter_two) {

	for (int i = 0; i < input_neurons->total_number_of_groups; i++) {

		AddSynapseGroup(-1 * i - 1, 
							postsynaptic_group_id, 
							connectivity_type,
							weight_range, 
							delay_range,
							stdp_on,
							connectivity_params,
							parameter,
							parameter_two);

	}

}


void Simulator::setup_network(bool temp_model_type) {

	printf("Setting Up Network...\n");
	clock_t initialise_network_start = clock();

	int threads_per_block_neurons = 512;
	int threads_per_block_synapses = 128;
	synapses->set_threads_per_block_and_blocks_per_grid(threads_per_block_synapses);
	neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);
	input_neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);

	// Provides order of magnitude speedup for Conductance (All to all atleast). 
	// Because all synapses contribute to current_injection on every iteration, having all threads in a block accessing only 1 or 2 positions in memory causes massive slowdown.
	// Randomising order of synapses means that each block is accessing a larger number of points in memory.
	// if (temp_model_type == 1) synapses->shuffle_synapses();

	neurons->allocate_device_pointers();
	synapses->allocate_device_pointers();
	input_neurons->allocate_device_pointers();

	input_neurons->generate_random_states();

	clock_t initialise_network_end = clock();
	float initialise_network_total_time = float(initialise_network_end - initialise_network_start) / CLOCKS_PER_SEC;
	printf("Network Setup. Time taken: %f\n", initialise_network_total_time);
	print_line_of_dashes_with_blank_lines_either_side();

}

void Simulator::setup_recording_electrodes() {

	printf("Setting Up Recording Electrodes...\n");
	clock_t setup_recording_electrodes_start = clock();

	recording_electrodes = new RecordingElectrodes(neurons);
	input_recording_electrodes = new RecordingElectrodes(input_neurons);
	recording_electrodes->initialise_device_pointers();
	recording_electrodes->initialise_host_pointers();
	input_recording_electrodes->initialise_device_pointers();
	input_recording_electrodes->initialise_host_pointers();

	clock_t setup_recording_electrodes_end = clock();
	float setup_recording_electrodes_total_time = float(setup_recording_electrodes_end - setup_recording_electrodes_start) / CLOCKS_PER_SEC;
	printf("Recording Electrodes Setup. Time taken: %f\n", setup_recording_electrodes_total_time);
	print_line_of_dashes_with_blank_lines_either_side();

}


void Simulator::Run(float total_time_per_epoch, int number_of_epochs, int temp_model_type, bool save_spikes, bool present_stimuli_in_random_order){

	// Check how many stimuli their are and do something about it:
	if (number_of_stimuli == 0){
		++number_of_stimuli;
		numEntries = (int*)realloc(numEntries, sizeof(int)*number_of_stimuli);
		numEntries[0] = 0;
	}
	
	if (number_of_epochs == 0) print_message_and_exit("Error. There must be at least one epoch.");

	// SEEDING
	srand(42);

	// STIMULUS ORDER (Put into function + variable)
	int stimuli_presentation_order[number_of_stimuli];
	for (int i = 0; i < number_of_stimuli; i++){
		stimuli_presentation_order[i] = i;
	}

	recording_electrodes->write_initial_synaptic_weights_to_file(synapses);


	begin_simulation_message(timestep, number_of_stimuli, number_of_epochs, save_spikes, present_stimuli_in_random_order, neurons->total_number_of_neurons, input_neurons->total_number_of_neurons, synapses->total_number_of_synapses);
	clock_t simulation_begin = clock();

	for (int epoch_number = 0; epoch_number < number_of_epochs; epoch_number++) {

		printf("\nStarting Epoch: %d\n", epoch_number);

		if (present_stimuli_in_random_order) {
			std::random_shuffle(&stimuli_presentation_order[0], &stimuli_presentation_order[number_of_stimuli]);
		}
		// Running through every Stimulus
		for (int stimulus_index = 0; stimulus_index < number_of_stimuli; stimulus_index++){

			// Reset the variables necessary
			neurons->reset_neurons();
			input_neurons->reset_neurons();
			synapses->reset_synapse_spikes();

			int number_of_timesteps_per_epoch = total_time_per_epoch / timestep;
			float current_time_in_seconds = 0.0f;
		
			for (int timestep_index = 0; timestep_index < number_of_timesteps_per_epoch; timestep_index++){
				
				current_time_in_seconds = float(timestep_index)*float(timestep);

				if (timestep_index % 10000 == 0) printf("current_time_in_seconds: %f\n", current_time_in_seconds);
				
				neurons->reset_current_injections();

				// Temporary seperation of izhikevich and Conductance per timestep instructions. Eventually hope to share as much execuation as possible between both models for generality
				if (temp_model_type == 0) temp_izhikevich_per_timestep_instructions(current_time_in_seconds);
				if (temp_model_type == 1) temp_conductance_per_timestep_instructions(current_time_in_seconds);

				// // Only save the spikes if necessary
				if (save_spikes){
					recording_electrodes->save_spikes_to_host(current_time_in_seconds, timestep_index, number_of_timesteps_per_epoch);
					input_recording_electrodes->save_spikes_to_host(current_time_in_seconds, timestep_index, number_of_timesteps_per_epoch);
				}
			}
		}
		#ifndef QUIETSTART
		clock_t simulation_mid = clock();
		if (save_spikes) {
			printf("Epoch %d, Complete.\n Running Time: %f\n Number of Spikes: %d\n\n", epoch_number, (float(simulation_mid-simulation_begin) / CLOCKS_PER_SEC), recording_electrodes->h_total_number_of_spikes);
			printf("Number of Input Spikes: %d\n\n", input_recording_electrodes->h_total_number_of_spikes);
		
		} else {
			printf("Epoch %d, Complete.\n Running Time: %f\n\n", epoch_number, (float(simulation_mid-simulation_begin) / CLOCKS_PER_SEC));
		}
		#endif
		// Output Spikes list after each epoch:
		// Only save the spikes if necessary
		if (save_spikes){
			recording_electrodes->write_spikes_to_file(neurons, epoch_number);
			input_recording_electrodes->write_spikes_to_file(input_neurons, epoch_number);
		}
	}
	
	// SIMULATION COMPLETE!
	#ifndef QUIETSTART
	// Finish the simulation and check time
	clock_t simulation_end = clock();
	float simulation_timed = float(simulation_end-simulation_begin) / CLOCKS_PER_SEC;
	printf("Simulation Complete! Time Elapsed: %f\n\n", simulation_timed);
	#endif

	recording_electrodes->save_network_state(synapses);

	delete recording_electrodes;
	delete input_recording_electrodes;

}


// Temporary seperation of izhikevich and Conductance per timestep instructions. Eventually hope to share as much execuation as possible between both models for generality
void Simulator::temp_izhikevich_per_timestep_instructions(float current_time_in_seconds) {

	// // If there are any spike generators
	// 	temp_test_generator->generupdate2_wrapper(current_time_in_seconds, timestep);
	
	synapses->check_for_synapse_spike_arrival(current_time_in_seconds);
	synapses->calculate_postsynaptic_current_injection(neurons, current_time_in_seconds);

	synapses->apply_ltd_to_synapse_weights(neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds);

	neurons->update_membrane_potentials(timestep);
	input_neurons->update_membrane_potentials(timestep);

	neurons->check_for_neuron_spikes(current_time_in_seconds);
	input_neurons->check_for_neuron_spikes(current_time_in_seconds);
					
	synapses->move_spikes_towards_synapses(neurons->d_last_spike_time_of_each_neuron, input_neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds);

	synapses->apply_ltp_to_synapse_weights(neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds);

}

void Simulator::temp_conductance_per_timestep_instructions(float current_time_in_seconds) {

	// Where generator->generupdate2_wrapper used to be
	
	synapses->check_for_synapse_spike_arrival(current_time_in_seconds);

	// Calculate I(t) from delta_g(t) and V(t)
	synapses->calculate_postsynaptic_current_injection(neurons, current_time_in_seconds);

	// Calculate g(t+delta_t) and delta_g(t)
	synapses->update_synaptic_conductances(timestep, current_time_in_seconds);
	
	// Calculate delta_g(t+delta_t) from C(t) and D(t)
	synapses->update_synaptic_efficacies_or_weights(neurons->d_recent_postsynaptic_activities_D, timestep, current_time_in_seconds, neurons->d_last_spike_time_of_each_neuron);

	// Calculate C(t+delta_t) from C(t)
	synapses->update_presynaptic_activities(timestep, current_time_in_seconds);

	// Calculate D(t+delta_t) from D(t)
	neurons->update_postsynaptic_activities(timestep, current_time_in_seconds);

	// Where synapses->LTD used to be

	// Caculate V(t+delta_t) from V(t) and I(t)
	neurons->update_membrane_potentials(timestep);
	input_neurons->update_membrane_potentials(timestep);

	// Check for NEURON_SPIKES(t+delta_t) from V(t+delta_t) and if so reset V(t+delta_t)
	neurons->check_for_neuron_spikes(current_time_in_seconds);
	input_neurons->check_for_neuron_spikes(current_time_in_seconds);
					
	synapses->move_spikes_towards_synapses(neurons->d_last_spike_time_of_each_neuron, input_neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds);

	// Where synapses->LTP used to be

}




// Spike Generator Spike Creation
// INPUT:
//		Population ID
//		Stimulus ID
//		Number of Neurons
//		Number of entries in our arrays
//		Array of generator indices (neuron IDs)
//		Corresponding array of the spike times for each instance
void Simulator::CreateGenerator(int popID, int stimulusid, int spikenumber, int* ids, float* spiketimes){
	// We have to ensure that we have created space for the current stimulus.
	if ((number_of_stimuli - 1) < stimulusid) {

		// Check what the difference is and quit if it is too high
		if ((stimulusid - (number_of_stimuli - 1)) > 1)	print_message_and_exit("Error: Stimuli not created in order.");

		// If it isn't greater than 1, make space!
		++number_of_stimuli;
		numEntries = (int*)realloc(numEntries, sizeof(int)*number_of_stimuli);
		genids = (int**)realloc(genids, sizeof(int*)*number_of_stimuli);
		gentimes = (float**)realloc(gentimes, sizeof(float*)*number_of_stimuli);
		// Initialize stuff
		genids[stimulusid] = NULL;
		gentimes[stimulusid] = NULL;
		numEntries[stimulusid] = 0;
	}
	// Spike generator populations are necessary
	// Create space for the new ids
	
	genids[stimulusid] = (int*)realloc(genids[stimulusid], 
								sizeof(int)*(spikenumber + numEntries[stimulusid]));
	gentimes[stimulusid] = (float*)realloc(gentimes[stimulusid], 
								sizeof(float)*(spikenumber + numEntries[stimulusid]));
	
	// Check where the neuron population starts
	int startnum = 0;
	if (popID > 0) {
		startnum = neurons->last_neuron_indices_for_each_group[popID-1];
	}
	
	// Assign the genid values according to how many neurons exist already
	for (int i = 0; i < spikenumber; i++){
		genids[stimulusid][numEntries[stimulusid]+i] = ids[i] + startnum;
		gentimes[stimulusid][numEntries[stimulusid]+i] = spiketimes[i];
	}
	// Increment the number of entries the generator population
	numEntries[stimulusid] += spikenumber;
	
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