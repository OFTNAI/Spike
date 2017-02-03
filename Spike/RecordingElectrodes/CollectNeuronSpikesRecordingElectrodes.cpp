#include "CollectNeuronSpikesRecordingElectrodes.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// CollectNeuronSpikesRecordingElectrodes Constructor
CollectNeuronSpikesRecordingElectrodes::CollectNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param) 
	: RecordingElectrodes(neurons_parameter, synapses_parameter, full_directory_name_for_simulation_data_files_param, prefix_string_param) {

	// Variables
	size_of_device_spike_store = 0;
	total_number_of_spikes_stored_on_host = 0;

	// Host Pointers
	collect_neuron_spikes_optional_parameters = new Collect_Neuron_Spikes_Optional_Parameters();
	neuron_ids_of_stored_spikes_on_host = nullptr;
	total_number_of_spikes_stored_on_device = nullptr;
	time_in_seconds_of_stored_spikes_on_host = nullptr;

	// Private Host Pointeres
	reset_neuron_ids = nullptr;
	reset_neuron_times = nullptr;

}


// CollectNeuronSpikesRecordingElectrodes Destructor
CollectNeuronSpikesRecordingElectrodes::~CollectNeuronSpikesRecordingElectrodes() {
  free(neuron_ids_of_stored_spikes_on_host);
  free(time_in_seconds_of_stored_spikes_on_host);
  free(total_number_of_spikes_stored_on_device);

  free(reset_neuron_ids);
  free(reset_neuron_times);
}


void CollectNeuronSpikesRecordingElectrodes::initialise_collect_neuron_spikes_recording_electrodes(Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters_param) {

	if (collect_neuron_spikes_optional_parameters_param != nullptr) {
		collect_neuron_spikes_optional_parameters = collect_neuron_spikes_optional_parameters_param;
	}

	size_of_device_spike_store = collect_neuron_spikes_optional_parameters->device_spike_store_size_multiple_of_total_neurons * neurons->total_number_of_neurons;

	allocate_pointers_for_spike_store();

}



void CollectNeuronSpikesRecordingElectrodes::allocate_pointers_for_spike_store() {

	total_number_of_spikes_stored_on_device = (int*)malloc(sizeof(int));
	total_number_of_spikes_stored_on_device[0] = 0;

	reset_neuron_ids = (int *)malloc(sizeof(int)*size_of_device_spike_store);
	reset_neuron_times = (float *)malloc(sizeof(float)*size_of_device_spike_store);
	for (int i=0; i < size_of_device_spike_store; i++){
		reset_neuron_ids[i] = -1;
		reset_neuron_times[i] = -1.0f;
	}
}

void CollectNeuronSpikesRecordingElectrodes::delete_and_reset_collected_spikes() {

	// Reset the spike store
	// Host values
	total_number_of_spikes_stored_on_host = 0;
	total_number_of_spikes_stored_on_device[0] = 0;
	// Free/Clear Device stuff
	// Reset the number on the device
    backend()->reset_state();

	// Free malloced host stuff
	free(neuron_ids_of_stored_spikes_on_host);
	free(time_in_seconds_of_stored_spikes_on_host);
	neuron_ids_of_stored_spikes_on_host = nullptr;
	time_in_seconds_of_stored_spikes_on_host = nullptr;
}




void CollectNeuronSpikesRecordingElectrodes::copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch) {

	if (((timestep_index % collect_neuron_spikes_optional_parameters->number_of_timesteps_per_device_spike_copy_check) == 0) || (timestep_index == (number_of_timesteps_per_epoch-1))){

		// Finally, we want to get the spikes back. Every few timesteps check the number of spikes:
                backend()->copy_spike_counts_to_front();

		// Ensure that we don't have too many
		if (total_number_of_spikes_stored_on_device[0] > size_of_device_spike_store){
			print_message_and_exit("Spike recorder has been overloaded! Reduce threshold.");
		}

		// Deal with them!
		if ((total_number_of_spikes_stored_on_device[0] >= (collect_neuron_spikes_optional_parameters->proportion_of_device_spike_store_full_before_copy * size_of_device_spike_store)) ||  (timestep_index == (number_of_timesteps_per_epoch - 1))){

			// Reallocate host spike arrays to accommodate for new device spikes.
			neuron_ids_of_stored_spikes_on_host = (int*)realloc(neuron_ids_of_stored_spikes_on_host, sizeof(int)*(total_number_of_spikes_stored_on_host + total_number_of_spikes_stored_on_device[0]));
			time_in_seconds_of_stored_spikes_on_host = (float*)realloc(time_in_seconds_of_stored_spikes_on_host, sizeof(float)*(total_number_of_spikes_stored_on_host + total_number_of_spikes_stored_on_device[0]));

			// Copy device spikes into correct host array location
                        backend()->copy_spikes_to_front();

			total_number_of_spikes_stored_on_host += total_number_of_spikes_stored_on_device[0];


			// Reset device spikes
                        backend()->reset_state();
 			total_number_of_spikes_stored_on_device[0] = 0;
		}
	}
}



void CollectNeuronSpikesRecordingElectrodes::write_spikes_to_file(int epoch_number, bool isTrained) {

	clock_t write_spikes_to_file_start = clock();

	// Get the names
	string phase = "";
	if (isTrained)
		phase = "Trained";
	else
		phase = "Untrained";
	
	string file_IDs = full_directory_name_for_simulation_data_files + prefix_string + "_SpikeIDs_" + phase + "_Epoch" + to_string(epoch_number);
	string file_Times = full_directory_name_for_simulation_data_files + prefix_string + "_SpikeTimes_" + phase + "_Epoch" + to_string(epoch_number);

//	// Append the clock to the file if flag
//	if (append_clock_to_filenames){ file = file + "t" + to_string(clock()) + "_"; }

	if (collect_neuron_spikes_optional_parameters->human_readable_storage){
		// Open the files
		ofstream spikeidfile, spiketimesfile;
		spikeidfile.open((file_IDs + ".txt"), ios::out | ios::binary);
		spiketimesfile.open((file_Times + ".txt"), ios::out | ios::binary);
		

		// Send the data
		for (int i = 0; i < total_number_of_spikes_stored_on_host; i++) {
			spikeidfile << to_string(neuron_ids_of_stored_spikes_on_host[i]) << endl;
			spiketimesfile << to_string(time_in_seconds_of_stored_spikes_on_host[i]) << endl;
		}

		// Close the files
		spikeidfile.close();
		spiketimesfile.close();
	} else {
		// Open the files
		ofstream spikeidfile, spiketimesfile;
		spikeidfile.open((file_IDs + ".bin"), ios::out | ios::binary);
		spiketimesfile.open((file_Times + ".bin"), ios::out | ios::binary);
		

		// Send the data
		spikeidfile.write((char *)neuron_ids_of_stored_spikes_on_host, total_number_of_spikes_stored_on_host*sizeof(int));
		spiketimesfile.write((char *)time_in_seconds_of_stored_spikes_on_host, total_number_of_spikes_stored_on_host*sizeof(float));

		// Close the files
		spikeidfile.close();
		spiketimesfile.close();
	}

	//delete_and_reset_recorded_spikes();

	clock_t write_spikes_to_file_end = clock();
	float write_spikes_to_file_total_time = float(write_spikes_to_file_end - write_spikes_to_file_start) / CLOCKS_PER_SEC;
	printf("Spikes written to file.\n Time taken: %f\n", write_spikes_to_file_total_time);
}


void CollectNeuronSpikesRecordingElectrodes::collect_spikes_for_timestep(float current_time_in_seconds) {
  backend()->collect_spikes_for_timestep(current_time_in_seconds);
}

SPIKE_MAKE_INIT_BACKEND(CollectNeuronSpikesRecordingElectrodes);
