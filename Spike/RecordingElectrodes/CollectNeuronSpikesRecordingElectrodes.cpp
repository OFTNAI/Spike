#include "CollectNeuronSpikesRecordingElectrodes.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
//CUDA #include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"
#include <string>
#include <time.h>
using namespace std;

// CollectNeuronSpikesRecordingElectrodes Constructor
CollectNeuronSpikesRecordingElectrodes::CollectNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param) 
	: RecordingElectrodes(neurons_parameter, synapses_parameter, full_directory_name_for_simulation_data_files_param, prefix_string_param) {

	// Variables
	size_of_device_spike_store = 0;
	h_total_number_of_spikes_stored_on_host = 0;

	// Host Pointers
	collect_neuron_spikes_optional_parameters = new Collect_Neuron_Spikes_Optional_Parameters();
	h_neuron_ids_of_stored_spikes_on_host = NULL;
	h_total_number_of_spikes_stored_on_device = NULL;
	h_time_in_seconds_of_stored_spikes_on_host = NULL;

	// Device Pointers
	d_neuron_ids_of_stored_spikes_on_device = NULL;
	d_total_number_of_spikes_stored_on_device = NULL;
	d_time_in_seconds_of_stored_spikes_on_device= NULL;

	// Private Host Pointeres
	reset_neuron_ids = NULL;
	reset_neuron_times = NULL;

}


// CollectNeuronSpikesRecordingElectrodes Destructor
CollectNeuronSpikesRecordingElectrodes::~CollectNeuronSpikesRecordingElectrodes() {

	free(h_neuron_ids_of_stored_spikes_on_host);
	free(h_time_in_seconds_of_stored_spikes_on_host);
	free(h_total_number_of_spikes_stored_on_device);

        /*CUDA
	CudaSafeCall(cudaFree(d_neuron_ids_of_stored_spikes_on_device));
	CudaSafeCall(cudaFree(d_total_number_of_spikes_stored_on_device));
	CudaSafeCall(cudaFree(d_time_in_seconds_of_stored_spikes_on_device));
        */

	free(reset_neuron_ids);
	free(reset_neuron_times);

}


void CollectNeuronSpikesRecordingElectrodes::initialise_collect_neuron_spikes_recording_electrodes(Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters_param) {

	if (collect_neuron_spikes_optional_parameters_param != NULL) {
		collect_neuron_spikes_optional_parameters = collect_neuron_spikes_optional_parameters_param;
	}

	size_of_device_spike_store = collect_neuron_spikes_optional_parameters->device_spike_store_size_multiple_of_total_neurons * neurons->total_number_of_neurons;

	allocate_pointers_for_spike_store();
	delete_and_reset_collected_spikes();

}



void CollectNeuronSpikesRecordingElectrodes::allocate_pointers_for_spike_store() {

	h_total_number_of_spikes_stored_on_device = (int*)malloc(sizeof(int));

        /*CUDA
	CudaSafeCall(cudaMalloc((void **)&d_neuron_ids_of_stored_spikes_on_device, sizeof(int)*size_of_device_spike_store));
	CudaSafeCall(cudaMalloc((void **)&d_time_in_seconds_of_stored_spikes_on_device, sizeof(float)*size_of_device_spike_store));
	CudaSafeCall(cudaMalloc((void **)&d_total_number_of_spikes_stored_on_device, sizeof(int)));
        */

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
	h_total_number_of_spikes_stored_on_host = 0;
	h_total_number_of_spikes_stored_on_device[0] = 0;
	// Free/Clear Device stuff
	// Reset the number on the device
        /*CUDA
	CudaSafeCall(cudaMemset(&(d_total_number_of_spikes_stored_on_device[0]), 0, sizeof(int)));
	CudaSafeCall(cudaMemset(d_neuron_ids_of_stored_spikes_on_device, -1, sizeof(int)*neurons->total_number_of_neurons));
	CudaSafeCall(cudaMemset(d_time_in_seconds_of_stored_spikes_on_device, -1.0f, sizeof(float)*neurons->total_number_of_neurons));
        */

	// Free malloced host stuff
	free(h_neuron_ids_of_stored_spikes_on_host);
	free(h_time_in_seconds_of_stored_spikes_on_host);
	h_neuron_ids_of_stored_spikes_on_host = NULL;
	h_time_in_seconds_of_stored_spikes_on_host = NULL;
}




void CollectNeuronSpikesRecordingElectrodes::copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch) {

	if (((timestep_index % collect_neuron_spikes_optional_parameters->number_of_timesteps_per_device_spike_copy_check) == 0) || (timestep_index == (number_of_timesteps_per_epoch-1))){

		// Finally, we want to get the spikes back. Every few timesteps check the number of spikes:
		//CUDA CudaSafeCall(cudaMemcpy(&(h_total_number_of_spikes_stored_on_device[0]), &(d_total_number_of_spikes_stored_on_device[0]), (sizeof(int)), cudaMemcpyDeviceToHost));

		// Ensure that we don't have too many
		if (h_total_number_of_spikes_stored_on_device[0] > size_of_device_spike_store){
			print_message_and_exit("Spike recorder has been overloaded! Reduce threshold.");
		}

		// Deal with them!
		if ((h_total_number_of_spikes_stored_on_device[0] >= (collect_neuron_spikes_optional_parameters->proportion_of_device_spike_store_full_before_copy * size_of_device_spike_store)) ||  (timestep_index == (number_of_timesteps_per_epoch - 1))){

			// Reallocate host spike arrays to accommodate for new device spikes.
			h_neuron_ids_of_stored_spikes_on_host = (int*)realloc(h_neuron_ids_of_stored_spikes_on_host, sizeof(int)*(h_total_number_of_spikes_stored_on_host + h_total_number_of_spikes_stored_on_device[0]));
			h_time_in_seconds_of_stored_spikes_on_host = (float*)realloc(h_time_in_seconds_of_stored_spikes_on_host, sizeof(float)*(h_total_number_of_spikes_stored_on_host + h_total_number_of_spikes_stored_on_device[0]));

			// Copy device spikes into correct host array location
                        /*CUDA
			CudaSafeCall(cudaMemcpy((void*)&h_neuron_ids_of_stored_spikes_on_host[h_total_number_of_spikes_stored_on_host], 
									d_neuron_ids_of_stored_spikes_on_device, 
									(sizeof(int)*h_total_number_of_spikes_stored_on_device[0]), 
									cudaMemcpyDeviceToHost));
			CudaSafeCall(cudaMemcpy((void*)&h_time_in_seconds_of_stored_spikes_on_host[h_total_number_of_spikes_stored_on_host], 
									d_time_in_seconds_of_stored_spikes_on_device, 
									sizeof(float)*h_total_number_of_spikes_stored_on_device[0], 
									cudaMemcpyDeviceToHost));
                        */

			h_total_number_of_spikes_stored_on_host += h_total_number_of_spikes_stored_on_device[0];


			// Reset device spikes
                        /*CUDA
			CudaSafeCall(cudaMemset(&(d_total_number_of_spikes_stored_on_device[0]), 0, sizeof(int)));
			CudaSafeCall(cudaMemcpy(d_neuron_ids_of_stored_spikes_on_device, reset_neuron_ids, sizeof(int)*size_of_device_spike_store, cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpy(d_time_in_seconds_of_stored_spikes_on_device, reset_neuron_times, sizeof(float)*size_of_device_spike_store, cudaMemcpyHostToDevice));
                        */
			h_total_number_of_spikes_stored_on_device[0] = 0;
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
		for (int i = 0; i < h_total_number_of_spikes_stored_on_host; i++) {
			spikeidfile << to_string(h_neuron_ids_of_stored_spikes_on_host[i]) << endl;
			spiketimesfile << to_string(h_time_in_seconds_of_stored_spikes_on_host[i]) << endl;
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
		spikeidfile.write((char *)h_neuron_ids_of_stored_spikes_on_host, h_total_number_of_spikes_stored_on_host*sizeof(int));
		spiketimesfile.write((char *)h_time_in_seconds_of_stored_spikes_on_host, h_total_number_of_spikes_stored_on_host*sizeof(float));

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
  /*CUDA
	collect_spikes_for_timestep_kernel<<<neurons->number_of_neuron_blocks_per_grid, neurons->threads_per_block>>>(neurons->d_last_spike_time_of_each_neuron,
														d_total_number_of_spikes_stored_on_device,
														d_neuron_ids_of_stored_spikes_on_device,
														d_time_in_seconds_of_stored_spikes_on_device,
														current_time_in_seconds,
														neurons->total_number_of_neurons);

	CudaCheckError();
  */
}



// Collect Spikes
/*CUDA
__global__ void collect_spikes_for_timestep_kernel(float* d_last_spike_time_of_each_neuron,
								int* d_total_number_of_spikes_stored_on_device,
								int* d_neuron_ids_of_stored_spikes_on_device,
								float* d_time_in_seconds_of_stored_spikes_on_device,
								float current_time_in_seconds,
								size_t total_number_of_neurons){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {

		// If a neuron has fired
		if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
			// Increase the number of spikes stored
			// NOTE: atomicAdd return value is actually original (atomic) value BEFORE incrementation!
			//		- So first value is actually 0 not 1!!!
			int i = atomicAdd(&d_total_number_of_spikes_stored_on_device[0], 1);
			__syncthreads();

			// In the location, add the id and the time
			d_neuron_ids_of_stored_spikes_on_device[i] = idx;
			d_time_in_seconds_of_stored_spikes_on_device[i] = current_time_in_seconds;
		}
		idx += blockDim.x * gridDim.x;
	}
	__syncthreads();
}
*/
