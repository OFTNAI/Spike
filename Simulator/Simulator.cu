#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <algorithm> // For random shuffle
#include <time.h>
#include <string>
#include <sys/stat.h>

#include "Simulator.h"

#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"
#include "../Helpers/RandomStateManager.h"

using namespace std;

// Constructors
Simulator::Simulator() {}


Simulator::Simulator(SpikingModel * spiking_model_param, Simulator_Options * simulator_options_param) {

	spiking_model = spiking_model_param;
	simulator_options = simulator_options_param;

	simulations_run_count = 0;

	full_directory_name_for_simulation_data_files = "output/"; // Put into struct!!

	TimerWithMessages * timer = new TimerWithMessages("Setting up recording electrodes...\n");

	if (simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool) {
		count_neuron_spikes_recording_electrodes = new CountNeuronSpikesRecordingElectrodes(spiking_model->spiking_neurons, spiking_model->spiking_synapses, full_directory_name_for_simulation_data_files, "Neurons");
		count_neuron_spikes_recording_electrodes->initialise_count_neuron_spikes_recording_electrodes();
	} else {
		count_neuron_spikes_recording_electrodes = NULL;
	}

	if (simulator_options->recording_electrodes_options->count_input_neuron_spikes_recording_electrodes_bool) {
		count_input_neuron_spikes_recording_electrodes = new CountNeuronSpikesRecordingElectrodes(spiking_model->input_spiking_neurons, spiking_model->spiking_synapses, full_directory_name_for_simulation_data_files, "Input_Neurons");
		count_input_neuron_spikes_recording_electrodes->initialise_count_neuron_spikes_recording_electrodes();
	} else {
		count_input_neuron_spikes_recording_electrodes = NULL;
	}

	if (simulator_options->recording_electrodes_options->collect_neuron_spikes_recording_electrodes_bool) {
		collect_neuron_spikes_recording_electrodes = new CollectNeuronSpikesRecordingElectrodes(spiking_model->spiking_neurons, spiking_model->spiking_synapses, full_directory_name_for_simulation_data_files, "Neurons");
		collect_neuron_spikes_recording_electrodes->initialise_collect_neuron_spikes_recording_electrodes(simulator_options->recording_electrodes_options->collect_neuron_spikes_optional_parameters);
	} else {
		collect_neuron_spikes_recording_electrodes = NULL;
	}

	if (simulator_options->recording_electrodes_options->collect_input_neuron_spikes_recording_electrodes_bool) {
		collect_input_neuron_spikes_recording_electrodes = new CollectNeuronSpikesRecordingElectrodes(spiking_model->input_spiking_neurons, spiking_model->spiking_synapses, full_directory_name_for_simulation_data_files, "Input_Neurons");
		collect_input_neuron_spikes_recording_electrodes->initialise_collect_neuron_spikes_recording_electrodes(simulator_options->recording_electrodes_options->collect_input_neuron_spikes_optional_parameters);
	} else {
		collect_input_neuron_spikes_recording_electrodes = NULL;
	}

	if (simulator_options->recording_electrodes_options->network_state_archive_recording_electrodes_bool) {
		network_state_archive_recording_electrodes->initialise_network_state_archive_recording_electrodes(simulator_options->recording_electrodes_options->network_state_archive_optional_parameters);
		network_state_archive_recording_electrodes = new NetworkStateArchiveRecordingElectrodes(spiking_model->spiking_neurons, spiking_model->spiking_synapses, full_directory_name_for_simulation_data_files, "Synapses");
	} else {
		network_state_archive_recording_electrodes = NULL;
	}


	timer->stop_timer_and_log_time_and_message("Recording electrodes setup.\n", true);

}

// Destructor
Simulator::~Simulator(){

	printf("Simulator Destructor\n");

	delete count_neuron_spikes_recording_electrodes;
	delete count_input_neuron_spikes_recording_electrodes;
	delete collect_neuron_spikes_recording_electrodes;
	delete collect_input_neuron_spikes_recording_electrodes;
	delete network_state_archive_recording_electrodes;

}

void Simulator::CreateDirectoryForSimulationDataFiles(string directory_name_for_simulation_data_files) {
	if (mkdir(("output/"+directory_name_for_simulation_data_files).c_str(),S_IRUSR | S_IWUSR | S_IXUSR | S_IRGRP | S_IWGRP | S_IXGRP | S_IROTH | S_IWOTH | S_IXOTH)==0)
		printf("\nDirectory created\n");
	else
		print_message_and_exit("\nERROR: You must set a different experiment name to avoid overwriting the results\n");
	full_directory_name_for_simulation_data_files = "output/"+directory_name_for_simulation_data_files+"/";
}


void Simulator::reset_all_recording_electrodes() {

	if (simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool) {
		count_neuron_spikes_recording_electrodes->reset_pointers_for_spike_count();
	}

	if (simulator_options->recording_electrodes_options->count_input_neuron_spikes_recording_electrodes_bool) {
		count_input_neuron_spikes_recording_electrodes->reset_pointers_for_spike_count();
	}

	if (simulator_options->recording_electrodes_options->collect_neuron_spikes_recording_electrodes_bool) {
		collect_neuron_spikes_recording_electrodes->delete_and_reset_collected_spikes();
	}

	if (simulator_options->recording_electrodes_options->collect_input_neuron_spikes_recording_electrodes_bool) {
		collect_input_neuron_spikes_recording_electrodes->delete_and_reset_collected_spikes();
	}

}



// void Simulator::RunSimulationToCountNeuronSpikes(float presentation_time_per_stimulus_per_epoch, bool collect_spikes, bool save_collected_spikes_and_states_to_file, SpikeAnalyser *spike_analyser, bool human_readable_storage, bool network_is_trained) {
// 	bool number_of_epochs = 1;
// 	bool apply_stdp_to_relevant_synapses = false;
// 	bool count_spikes_per_neuron = true;
// 	int stimulus_presentation_order_seed = 0; // Shouldn't be needed if stimuli presentation not random
// 	Stimuli_Presentation_Struct * simulator_options->stimuli_presentation_options = new Stimuli_Presentation_Struct();
// 	// simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
// 	simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI;
// 	simulator_options->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
// 	simulator_options->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;
	
// 	if (!network_is_trained)
// 		recording_electrodes->write_initial_synaptic_weights_to_file(spiking_model->spiking_synapses, human_readable_storage);
	
// 	RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, collect_spikes, save_collected_spikes_and_states_to_file, apply_stdp_to_relevant_synapses, count_spikes_per_neuron, simulator_options->stimuli_presentation_options, stimulus_presentation_order_seed, spike_analyser,human_readable_storage,network_is_trained);
	
// 	if (network_is_trained)
// 		recording_electrodes->write_network_state_to_file(spiking_model->spiking_synapses, human_readable_storage);

// }


// void Simulator::RunSimulationToCollectEvents(float presentation_time_per_stimulus_per_epoch, bool network_is_trained) {
// 	bool number_of_epochs = 1;
// 	bool apply_stdp_to_relevant_synapses = false;
// 	bool count_spikes_per_neuron = true;
// 	int stimulus_presentation_order_seed = 0; // Shouldn't be needed if stimuli presentation not random
// 	Stimuli_Presentation_Struct * simulator_options->stimuli_presentation_options = new Stimuli_Presentation_Struct();
// 	// simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS;
// 	simulator_options->stimuli_presentation_options->presentation_format = PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI;
// 	simulator_options->stimuli_presentation_options->object_order = OBJECT_ORDER_ORIGINAL;
// 	simulator_options->stimuli_presentation_options->transform_order = TRANSFORM_ORDER_ORIGINAL;

// 	bool collect_spikes = false;
// 	bool save_collected_spikes_and_states_to_file = false;

// 	SpikeAnalyser * spike_analyser = NULL;
// 	bool human_readable_storage = false;
	
// 	RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, collect_spikes, save_collected_spikes_and_states_to_file, apply_stdp_to_relevant_synapses, count_spikes_per_neuron, simulator_options->stimuli_presentation_options, stimulus_presentation_order_seed, spike_analyser, human_readable_storage,network_is_trained);
	
// }

// void Simulator::RunSimulationToTrainNetwork(float presentation_time_per_stimulus_per_epoch, int number_of_epochs, Stimuli_Presentation_Struct * simulator_options->stimuli_presentation_options, int stimulus_presentation_order_seed) {

// 	bool apply_stdp_to_relevant_synapses = true;
// 	bool count_spikes_per_neuron = false;
// 	bool collect_spikes = false;
// 	bool save_collected_spikes_and_states_to_file = false;

// 	RunSimulation(presentation_time_per_stimulus_per_epoch, number_of_epochs, collect_spikes, save_collected_spikes_and_states_to_file, apply_stdp_to_relevant_synapses, count_spikes_per_neuron, simulator_options->stimuli_presentation_options, stimulus_presentation_order_seed, NULL, false, false);
// }



void Simulator::RunSimulation(SpikeAnalyser *spike_analyser) {

	// check_for_epochs_and_begin_simulation_message(spiking_model->timestep, spiking_model->input_spiking_neurons->total_number_of_input_stimuli, number_of_epochs, collect_spikes, save_collected_spikes_and_states_to_file, spiking_model->spiking_neurons->total_number_of_neurons, spiking_model->input_spiking_neurons->total_number_of_neurons, spiking_model->spiking_synapses->total_number_of_synapses);
	// Should print something about simulator_options->stimuli_presentation_options as old stuff removed from check_for_epochs...
	TimerWithMessages * simulation_timer = new TimerWithMessages();

	// Set seed for stimulus presentation order
	srand(simulator_options->run_simulation_general_options->stimulus_presentation_order_seed);


	reset_all_recording_electrodes();

	if (simulator_options->file_storage_options->write_initial_synaptic_weights_to_file_bool) {
	
		network_state_archive_recording_electrodes->write_initial_synaptic_weights_to_file();
	
	}

	for (int epoch_number = 0; epoch_number < simulator_options->run_simulation_general_options->number_of_epochs; epoch_number++) {

		TimerWithMessages * epoch_timer = new TimerWithMessages();
		printf("Starting Epoch: %d\n", epoch_number);

		spiking_model->reset_model_activities();

		float current_time_in_seconds = 0.0f;

		int* stimuli_presentation_order = setup_stimuli_presentation_order();
		for (int stimulus_index = 0; stimulus_index < spiking_model->input_spiking_neurons->total_number_of_input_stimuli; stimulus_index++) {

			if (simulator_options->stimuli_presentation_options->reset_current_time_between_each_stimulus) current_time_in_seconds = 0.0f; // For GeneratorInputSpikingNeurons?

			perform_pre_stimulus_presentation_instructions(stimuli_presentation_order[stimulus_index]);

			int number_of_timesteps_per_stimulus_per_epoch = simulator_options->run_simulation_general_options->presentation_time_per_stimulus_per_epoch / spiking_model->timestep;
		

			for (int timestep_index = 0; timestep_index < number_of_timesteps_per_stimulus_per_epoch; timestep_index++){
				
				spiking_model->perform_per_timestep_model_instructions(current_time_in_seconds, simulator_options->run_simulation_general_options->apply_stdp_to_relevant_synapses);

				perform_per_timestep_recording_electrode_instructions(current_time_in_seconds, timestep_index, number_of_timesteps_per_stimulus_per_epoch);

				current_time_in_seconds += float(spiking_model->timestep);

			}

			perform_post_stimulus_presentation_instructions(spike_analyser);
			
		}

		perform_post_epoch_instructions(epoch_number, epoch_timer);
		
	}
	
	perform_end_of_simulation_instructions(simulation_timer);
	
}


int* Simulator::setup_stimuli_presentation_order() {

	int total_number_of_input_stimuli = spiking_model->input_spiking_neurons->total_number_of_input_stimuli;
	int total_number_of_objects = spiking_model->input_spiking_neurons->total_number_of_objects;
	int total_number_of_transformations_per_object = spiking_model->input_spiking_neurons->total_number_of_transformations_per_object;
	
	int* stimuli_presentation_order = (int*)malloc(total_number_of_input_stimuli*sizeof(int));

	// From InputSpikingNeurons
	
	for (int i = 0; i < total_number_of_input_stimuli; i++){
		stimuli_presentation_order[i] = i;
	}

	switch (simulator_options->stimuli_presentation_options->presentation_format) {

		case PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS: case PRESENTATION_FORMAT_RANDOM_NO_RESET: {
			std::random_shuffle(&stimuli_presentation_order[0], &stimuli_presentation_order[total_number_of_input_stimuli]);
			break;
		}

		case PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS: case PRESENTATION_FORMAT_OBJECT_BY_OBJECT_NO_RESET: {
			
			int* object_order_indices = (int*)malloc(total_number_of_objects * sizeof(int));

			for (int object_index = 0; object_index < total_number_of_objects; object_index++) {
				object_order_indices[object_index] = object_index;			
			}

			switch (simulator_options->stimuli_presentation_options->object_order) {
		
				case OBJECT_ORDER_ORIGINAL:

					break;

				case OBJECT_ORDER_RANDOM:
					std::random_shuffle(&object_order_indices[0], &object_order_indices[total_number_of_objects]);
					break;

			}

			int* transform_order_indices = (int*)malloc(total_number_of_transformations_per_object * sizeof(int));
			for (int transform_index = 0; transform_index < total_number_of_transformations_per_object; transform_index++) {
				transform_order_indices[transform_index] = transform_index;			
			}

			for (int object_index = 0; object_index < total_number_of_objects; object_index++) {
				
				if (simulator_options->stimuli_presentation_options->transform_order == TRANSFORM_ORDER_RANDOM) std::random_shuffle(&transform_order_indices[0], &transform_order_indices[total_number_of_transformations_per_object]);

				for (int transform_index = 0; transform_index < total_number_of_transformations_per_object; transform_index++) {
					stimuli_presentation_order[object_index * total_number_of_transformations_per_object + transform_index] = object_order_indices[object_index] * total_number_of_transformations_per_object + transform_order_indices[transform_index]; 
				}					
			}

			break;

		}

		default:
			break;
	}

	return stimuli_presentation_order;
}



void Simulator::perform_per_timestep_recording_electrode_instructions(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_stimulus_per_epoch) {

	// Do various recording electrode operations


	// JI PSEUDO CODE FOR COLLECTING EVENTS START

	// if synapse_spike_arrived

	// 	int neuron_spike_count_for_postsynaptic_neuron = recording_electrodes->d_per_neuron_spike_counts[postsynaptic_neuron_id];

	// 	if (current_time_in_seconds > (d_ordered_spike_times_data[d_beginning_spike_time_int_indices_per_neuron[neuron_index] + neuron_spike_count_for_postsynaptic_neuron] - window) {
	// 		d_events_as_bools_per_neuron_and_spike_data[d_beginning_event_bool_indices_per_neuron + neuron_spike_count_for_postsynaptic_neuron * neurons->per_neuron_afferent_synapse_count[postsynaptic_neuron_index] + synapses->d_synapse_postsynaptic_neuron_count_index[synapse_index]] = true;
	// }


	// JI PSEUDO CODE FOR COLLECTING EVENTS END



	if (simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool) {
	
		count_neuron_spikes_recording_electrodes->add_spikes_to_per_neuron_spike_count(current_time_in_seconds);
	
	}

	if (simulator_options->recording_electrodes_options->count_input_neuron_spikes_recording_electrodes_bool) {
	
		count_input_neuron_spikes_recording_electrodes->add_spikes_to_per_neuron_spike_count(current_time_in_seconds);
	
	}

	if (simulator_options->recording_electrodes_options->collect_neuron_spikes_recording_electrodes_bool){

		collect_neuron_spikes_recording_electrodes->collect_spikes_for_timestep(current_time_in_seconds);
		collect_neuron_spikes_recording_electrodes->copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time_in_seconds, timestep_index, number_of_timesteps_per_stimulus_per_epoch );
	
	}

	if (simulator_options->recording_electrodes_options->collect_input_neuron_spikes_recording_electrodes_bool) {
		
		collect_input_neuron_spikes_recording_electrodes->collect_spikes_for_timestep(current_time_in_seconds);
		collect_input_neuron_spikes_recording_electrodes->copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(current_time_in_seconds, timestep_index, number_of_timesteps_per_stimulus_per_epoch );

	}

}


void Simulator::perform_pre_stimulus_presentation_instructions(int stimulus_index) {

	printf("Stimulus Index: %d\n", stimulus_index);
	printf("simulator_options->stimuli_presentation_options->presentation_format: %d\n", simulator_options->stimuli_presentation_options->presentation_format);

	switch (simulator_options->stimuli_presentation_options->presentation_format) {
		case PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_STIMULI: case PRESENTATION_FORMAT_RANDOM_RESET_BETWEEN_EACH_STIMULUS:
		{
			spiking_model->reset_model_activities();

			break;
		}
		case PRESENTATION_FORMAT_OBJECT_BY_OBJECT_RESET_BETWEEN_OBJECTS:
		{
			bool stimulus_is_new_object = spiking_model->input_spiking_neurons->stimulus_is_new_object_for_object_by_object_presentation(stimulus_index);
			(stimulus_is_new_object) ? printf("stimulus_is_new_object\n") : printf("stimulus_is_NOT_new_object\n");

			if (stimulus_is_new_object) {
				spiking_model->reset_model_activities();
			}

			break;
		}
		default:
			break;

	}


	spiking_model->input_spiking_neurons->current_stimulus_index = stimulus_index;
	spiking_model->input_spiking_neurons->reset_neuron_activities();


}



void Simulator::perform_post_stimulus_presentation_instructions(SpikeAnalyser * spike_analyser) {

	if (simulator_options->recording_electrodes_options->count_neuron_spikes_recording_electrodes_bool && spike_analyser) {

		spike_analyser->store_spike_counts_for_stimulus_index(spiking_model->input_spiking_neurons->current_stimulus_index, count_neuron_spikes_recording_electrodes->d_per_neuron_spike_counts);
		count_neuron_spikes_recording_electrodes->reset_pointers_for_spike_count();

	}

}


void Simulator::perform_post_epoch_instructions(int epoch_number, TimerWithMessages * epoch_timer) {

	printf("Epoch %d, Complete.\n", epoch_number);
	epoch_timer->stop_timer_and_log_time_and_message(" ", true);
	
	if (simulator_options->recording_electrodes_options->collect_neuron_spikes_recording_electrodes_bool) printf(" Number of Spikes: %d\n", collect_neuron_spikes_recording_electrodes->h_total_number_of_spikes_stored_on_host);
	if (simulator_options->recording_electrodes_options->collect_input_neuron_spikes_recording_electrodes_bool) printf(" Number of Input Spikes: %d\n", collect_input_neuron_spikes_recording_electrodes->h_total_number_of_spikes_stored_on_host);

	if (simulator_options->file_storage_options->save_recorded_neuron_spikes_to_file) collect_neuron_spikes_recording_electrodes->write_spikes_to_file(epoch_number, simulator_options->file_storage_options->network_is_trained);
	if (simulator_options->file_storage_options->save_recorded_input_neuron_spikes_to_file) collect_input_neuron_spikes_recording_electrodes->write_spikes_to_file(epoch_number, simulator_options->file_storage_options->network_is_trained);

}


void Simulator::perform_end_of_simulation_instructions(TimerWithMessages * simulation_timer) {

	simulation_timer->stop_timer_and_log_time_and_message("Simulation Complete!", true);

	if (simulator_options->recording_electrodes_options->network_state_archive_recording_electrodes_bool) {

		network_state_archive_recording_electrodes->write_network_state_to_file();

	}

	simulations_run_count++;

}