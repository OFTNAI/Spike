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
#include "../Helpers/RandomStateManager.h"


// Constructor
Simulator::Simulator(){

	// Default parameters

	recording_electrodes = NULL;
	input_recording_electrodes = NULL;

	// Default low fidelity spike storage
	high_fidelity_spike_storage = false;
	
	// #ifndef QUIETSTART
	// 	print_line_of_dashes_with_blank_lines_either_side();
	// 	printf("Welcome to the SPIKE.\n");
	// 	print_line_of_dashes_with_blank_lines_either_side();
	// 	fflush(stdout);
	// #endif
		
}


// Destructor
Simulator::~Simulator(){

}


void Simulator::SetSpikingModel(SpikingModel * spiking_model_parameter) {
	spiking_model = spiking_model_parameter;
}



void Simulator::setup_recording_electrodes_for_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param) {

	TimerWithMessages * timer = new TimerWithMessages("Setting up recording electrodes for neurons...\n");

	recording_electrodes = new RecordingElectrodes(spiking_model->spiking_neurons, "Neurons", number_of_timesteps_per_device_spike_copy_check_param, device_spike_store_size_multiple_of_total_neurons_param, proportion_of_device_spike_store_full_before_copy_param);
	
	recording_electrodes->allocate_pointers_for_spike_store();
	recording_electrodes->reset_pointers_for_spike_store();

	recording_electrodes->allocate_pointers_for_spike_count();
	recording_electrodes->reset_pointers_for_spike_count();

	timer->stop_timer_and_log_time_and_message("Recording Electrodes Setup For Neurons.", true);
}


void Simulator::setup_recording_electrodes_for_input_neurons(int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param) {

	TimerWithMessages * timer = new TimerWithMessages("Setting Up recording electrodes for input neurons...\n");

	input_recording_electrodes = new RecordingElectrodes(spiking_model->input_spiking_neurons, "Input_Spiking_Neurons", number_of_timesteps_per_device_spike_copy_check_param, device_spike_store_size_multiple_of_total_neurons_param, proportion_of_device_spike_store_full_before_copy_param);
	
	input_recording_electrodes->allocate_pointers_for_spike_store();
	input_recording_electrodes->reset_pointers_for_spike_store();

	input_recording_electrodes->allocate_pointers_for_spike_count();
	input_recording_electrodes->reset_pointers_for_spike_count();

	timer->stop_timer_and_log_time_and_message("Recording Electrodes Setup For Input Neurons.", true);
}


void Simulator::RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool record_spikes, bool save_recorded_spikes_and_states_to_file, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained) {
	bool number_of_epochs = 1;
	bool apply_stdp_to_relevant_synapses = false;
	bool count_spikes_per_neuron = true;
	int stimulus_presentation_order_seed = 0; // Shouldn't be needed if stimuli presentation not random
	Stimuli_Presentation_Struct * stimuli_presentation_params = new Stimuli_Presentation_Struct();
	// stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
	stimuli_presentation_params->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI;
	stimuli_presentation_params->object_order = OBJECT_ORDER_ORIGINAL;
	stimuli_presentation_params->transform_order = TRANSFORM_ORDER_ORIGINAL;
	
	if (!isTrained)
		recording_electrodes->write_initial_synaptic_weights_to_file(spiking_model->spiking_synapses, human_readable_storage);
	
	RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, record_spikes, save_recorded_spikes_and_states_to_file, apply_stdp_to_relevant_synapses, count_spikes_per_neuron, stimuli_presentation_params, stimulus_presentation_order_seed, spike_analyser,human_readable_storage,isTrained);
	
	if (isTrained)
		recording_electrodes->write_network_state_to_file(spiking_model->spiking_synapses, human_readable_storage);

}

void Simulator::RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed) {

	bool apply_stdp_to_relevant_synapses = true;
	bool count_spikes_per_neuron = false;
	bool record_spikes = false;
	bool save_recorded_spikes_and_states_to_file = false;

	RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, record_spikes, save_recorded_spikes_and_states_to_file, apply_stdp_to_relevant_synapses, count_spikes_per_neuron, stimuli_presentation_params, stimulus_presentation_order_seed, NULL, false, false);
}



void Simulator::RunSimulation(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, bool record_spikes, bool save_recorded_spikes_and_states_to_file, bool apply_stdp_to_relevant_synapses, bool count_spikes_per_neuron, Stimuli_Presentation_Struct * stimuli_presentation_params, int stimulus_presentation_order_seed, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool isTrained){

	check_for_epochs_and_begin_simulation_message(spiking_model->timestep, spiking_model->input_spiking_neurons->total_number_of_input_stimuli, number_of_epochs, record_spikes, save_recorded_spikes_and_states_to_file, spiking_model->spiking_neurons->total_number_of_neurons, spiking_model->input_spiking_neurons->total_number_of_neurons, spiking_model->spiking_synapses->total_number_of_synapses);
	// Should print something about stimuli_presentation_params as old stuff removed from check_for_epochs...
	TimerWithMessages * simulation_timer = new TimerWithMessages();

	// Set seed for stimulus presentation order
	srand(stimulus_presentation_order_seed);


	if (recording_electrodes) recording_electrodes->delete_and_reset_recorded_spikes();

	for (int epoch_number = 0; epoch_number < number_of_epochs; epoch_number++) {
	
		TimerWithMessages * epoch_timer = new TimerWithMessages();
		printf("Starting Epoch: %d\n", epoch_number);

		spiking_model->spiking_neurons->reset_neuron_activities();
		spiking_model->spiking_synapses->reset_synapse_activities();
		spiking_model->stdp_rule->reset_STDP_activities();

		float current_time_in_seconds = 0.0f;

		int* stimuli_presentation_order = spiking_model->input_spiking_neurons->setup_stimuli_presentation_order(stimuli_presentation_params);
		for (int stimulus_index = 0; stimulus_index < spiking_model->input_spiking_neurons->total_number_of_input_stimuli; stimulus_index++) {

			printf("Stimulus: %d, Current time in seconds: %1.2f\n", stimuli_presentation_order[stimulus_index], current_time_in_seconds);
			printf("stimuli_presentation_params->presentation_format: %d\n", stimuli_presentation_params->presentation_format);

			switch (stimuli_presentation_params->presentation_format) {
				case PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI: case PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS:
				{
					spiking_model->spiking_neurons->reset_neuron_activities();
					spiking_model->input_spiking_neurons->reset_neuron_activities();
					spiking_model->spiking_synapses->reset_synapse_activities();
					spiking_model->stdp_rule->reset_STDP_activities();

					break;
				}
				case PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS:
				{
					bool stimulus_is_new_object = spiking_model->input_spiking_neurons->stimulus_is_new_object_for_object_by_object_presentation(stimulus_index);
					(stimulus_is_new_object) ? printf("stimulus_is_new_object\n") : printf("stimulus_is_NOT_new_object\n");

					if (stimulus_is_new_object) {
						spiking_model->spiking_neurons->reset_neuron_activities();
						spiking_model->input_spiking_neurons->reset_neuron_activities();
						spiking_model->spiking_synapses->reset_synapse_activities();
						spiking_model->stdp_rule->reset_STDP_activities();
					}

					break;
				}
				default:
					break;

			}


			// These statements have been placed in this order for the Spike Generator Neurons so that they can set up for the next timestep
			spiking_model->input_spiking_neurons->current_stimulus_index = stimuli_presentation_order[stimulus_index];
			spiking_model->input_spiking_neurons->reset_neuron_activities();

			int number_of_timesteps_per_stimulus_per_epoch = presentation_time_per_stimulus_per_epoch / spiking_model->timestep;
		
			for (int timestep_index = 0; timestep_index < number_of_timesteps_per_stimulus_per_epoch; timestep_index++){
				
				spiking_model->spiking_neurons->reset_current_injections();

				// Carry out the per-timestep computations			
				per_timestep_instructions(current_time_in_seconds, apply_stdp_to_relevant_synapses);

				if (count_spikes_per_neuron) {
					if (recording_electrodes) recording_electrodes->add_spikes_to_per_neuron_spike_count(current_time_in_seconds);
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

				current_time_in_seconds += float(spiking_model->timestep);

			}

			if (count_spikes_per_neuron) {
				if (spike_analyser) {
					spike_analyser->store_spike_counts_for_stimulus_index(spiking_model->input_spiking_neurons->current_stimulus_index, recording_electrodes->d_per_neuron_spike_counts);
					if (recording_electrodes) recording_electrodes->reset_pointers_for_spike_count();
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
		if (record_spikes && save_recorded_spikes_and_states_to_file){
			printf("Write to file\n");
			if (recording_electrodes) recording_electrodes->write_spikes_to_file(epoch_number, human_readable_storage, isTrained);
			if (input_recording_electrodes) input_recording_electrodes->write_spikes_to_file(epoch_number, human_readable_storage,isTrained);
		}
	}
	
	// SIMULATION COMPLETE!
	#ifndef QUIETSTART
	simulation_timer->stop_timer_and_log_time_and_message("Simulation Complete!", true);
	#endif

//	recording_electrodes->write_network_state_to_file(spiking_model->spiking_synapses, human_readable_storage);

	// delete recording_electrodes;
	// delete input_recording_electrodes;

}


void Simulator::per_timestep_instructions(float current_time_in_seconds, bool apply_stdp_to_relevant_synapses){

	spiking_model->spiking_neurons->check_for_neuron_spikes(current_time_in_seconds, spiking_model->timestep);
	spiking_model->input_spiking_neurons->check_for_neuron_spikes(current_time_in_seconds, spiking_model->timestep);

	spiking_model->spiking_synapses->interact_spikes_with_synapses(spiking_model->spiking_neurons, spiking_model->input_spiking_neurons, current_time_in_seconds, spiking_model->timestep);

	spiking_model->spiking_synapses->calculate_postsynaptic_current_injection(spiking_model->spiking_neurons, current_time_in_seconds, spiking_model->timestep);

	if (apply_stdp_to_relevant_synapses){
		spiking_model->stdp_rule->Run_STDP(spiking_model->spiking_neurons->d_last_spike_time_of_each_neuron, current_time_in_seconds, spiking_model->timestep);
	}

	spiking_model->spiking_neurons->update_membrane_potentials(spiking_model->timestep, current_time_in_seconds);
	spiking_model->input_spiking_neurons->update_membrane_potentials(spiking_model->timestep, current_time_in_seconds);

}