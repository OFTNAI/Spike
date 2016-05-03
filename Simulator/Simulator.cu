// 	Simulator Class
// 	Simulator.cu

//	Authors: Nasir Ahmad (7/12/2015), James Isbister (23/3/2016)

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm> // For random shuffle
#include <time.h>

#include "Simulator.h"
#include "../RecordingElectrodes/RecordingElectrodes.h"
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
	
	#ifndef QUIETSTART
		// Say Hi to the user:
		printf("\nWelcome to the SPIKE.\n\n");
		printf("Setting up Populations and Synapses: \n\n");
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
							parameter,
							parameter_two);
}




void Simulator::Run(float total_time_per_epoch, int number_of_epochs, int temp_model_type, bool save_spikes, bool present_stimuli_in_random_order){

	begin_simulation_message(timestep, number_of_stimuli, number_of_epochs, save_spikes, present_stimuli_in_random_order, neurons->total_number_of_neurons, synapses->total_number_of_synapses);

	// Check how many stimuli their are and do something about it:
	if (number_of_stimuli == 0){
		++number_of_stimuli;
		numEntries = (int*)realloc(numEntries, sizeof(int)*number_of_stimuli);
		numEntries[0] = 0;
	}
	
	if (number_of_epochs == 0) print_message_and_exit("Error. There must be at least one epoch.");

	GeneratorSpikingNeurons * temp_test_generator = new GeneratorSpikingNeurons();
	RecordingElectrodes * recording_electrodes = new RecordingElectrodes(neurons);
	RecordingElectrodes * input_recording_electrodes = new RecordingElectrodes(input_neurons);

	int threads_per_block_neurons = 512;
	int threads_per_block_synapses = 512;
	synapses->set_threads_per_block_and_blocks_per_grid(threads_per_block_synapses);
	neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);
	input_neurons->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);

	// Provides order of magnitude speedup for LIF (All to all atleast). 
	// Because all synapses contribute to current_injection on every iteration, having all threads in a block accessing only 1 or 2 positions in memory causing massive slowdown.
	// Randomising order of synapses means that each block is accessing a larger number of points in memory.
	if (temp_model_type == 1) synapses->shuffle_synapses();
	// synapses->shuffle_synapses();

	neurons->allocate_device_pointers();
	synapses->allocate_device_pointers();
	input_neurons->allocate_device_pointers();

	recording_electrodes->initialise_device_pointers();
	recording_electrodes->initialise_host_pointers();
	input_recording_electrodes->initialise_device_pointers();
	input_recording_electrodes->initialise_host_pointers();

	// SEEDING
	srand(42);

	// STIMULUS ORDER (Put into function + variable)
	int stimuli_presentation_order[number_of_stimuli];
	for (int i = 0; i < number_of_stimuli; i++){
		stimuli_presentation_order[i] = i;
	}

	recording_electrodes->write_initial_synaptic_weights_to_file(synapses);
	
	input_neurons->generate_random_states();


	clock_t begin = clock();

	for (int epoch_number = 0; epoch_number < number_of_epochs; epoch_number++) {

		if (present_stimuli_in_random_order) {
			std::random_shuffle(&stimuli_presentation_order[0], &stimuli_presentation_order[number_of_stimuli]);
		}
		// Running through every Stimulus
		for (int stimulus_index = 0; stimulus_index < number_of_stimuli; stimulus_index++){
			// Get the presentation position:
			int present = stimuli_presentation_order[stimulus_index];
			// Get the number of entries for this specific stimulus
			size_t numEnts = numEntries[present];
			if (numEnts > 0){
				temp_test_generator->initialise_device_pointers_for_ents(numEnts, present);
				temp_test_generator->set_threads_per_block_and_blocks_per_grid(threads_per_block_neurons);
			}
			// Reset the variables necessary
			neurons->reset_neurons();
			input_neurons->reset_neurons();
			synapses->reset_synapse_spikes();

			int number_of_timesteps_per_epoch = total_time_per_epoch / timestep;
			float current_time_in_seconds = 0.0f;
		
			for (int timestep_index = 0; timestep_index < number_of_timesteps_per_epoch; timestep_index++){
				
				current_time_in_seconds = float(timestep_index)*float(timestep);
				
				neurons->reset_current_injections();

				// Temporary seperation of izhikevich and LIF per timestep instructions. Eventually hope to share as much execuation as possible between both models for generality
				if (temp_model_type == 0) temp_izhikevich_per_timestep_instructions(current_time_in_seconds);
				if (temp_model_type == 1) temp_lif_per_timestep_instructions(current_time_in_seconds);

				// // Only save the spikes if necessary
				if (save_spikes){
					recording_electrodes->save_spikes_to_host(current_time_in_seconds, timestep_index, number_of_timesteps_per_epoch);
					input_recording_electrodes->save_spikes_to_host(current_time_in_seconds, timestep_index, number_of_timesteps_per_epoch);
				}
			}
			if (numEnts > 0){
				// CudaSafeCall(cudaFree(d_genids));
				// CudaSafeCall(cudaFree(d_gentimes));
			}
		}
		#ifndef QUIETSTART
		clock_t mid = clock();
		if (save_spikes) {
			printf("Epoch %d, Complete.\n Running Time: %f\n Number of Spikes: %d\n\n", epoch_number, (float(mid-begin) / CLOCKS_PER_SEC), recording_electrodes->h_total_number_of_spikes);
			printf("Number of Input Spikes: %d\n\n", input_recording_electrodes->h_total_number_of_spikes);
		
		} else {
			printf("Epoch %d, Complete.\n Running Time: %f\n\n", epoch_number, (float(mid-begin) / CLOCKS_PER_SEC));
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
	clock_t end = clock();
	float timed = float(end-begin) / CLOCKS_PER_SEC;
	printf("Simulation Complete! Time Elapsed: %f\n\n", timed);
	#endif

	recording_electrodes->save_network_state(synapses);

	delete recording_electrodes;
	delete input_recording_electrodes;

}


// Temporary seperation of izhikevich and LIF per timestep instructions. Eventually hope to share as much execuation as possible between both models for generality
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

void Simulator::temp_lif_per_timestep_instructions(float current_time_in_seconds) {

	// // If there are any spike generators
	// 	temp_test_generator->generupdate2_wrapper(current_time_in_seconds, timestep);
	
	synapses->check_for_synapse_spike_arrival(current_time_in_seconds);
	synapses->calculate_postsynaptic_current_injection(neurons, current_time_in_seconds);
	synapses->update_synaptic_conductances(timestep, current_time_in_seconds);


	// synapses->apply_ltd_to_synapse_weights(neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds);

	neurons->update_membrane_potentials(timestep);
	input_neurons->update_membrane_potentials(timestep);

	neurons->check_for_neuron_spikes(current_time_in_seconds);
	input_neurons->check_for_neuron_spikes(current_time_in_seconds);
					
	synapses->move_spikes_towards_synapses(neurons->d_last_spike_time_of_each_neuron, input_neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds);

	// synapses->apply_ltp_to_synapse_weights(neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds);

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