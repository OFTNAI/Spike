//	RecordingElectrodes Class C++
//	RecordingElectrodes.cu
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#include "RecordingElectrodes.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"
#include <string>
#include <time.h>
using namespace std;

// RecordingElectrodes Constructor
RecordingElectrodes::RecordingElectrodes(SpikingNeurons * neurons_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param, int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param) {

	neurons = neurons_parameter;
	full_directory_name_for_simulation_data_files = full_directory_name_for_simulation_data_files_param;
	prefix_string = prefix_string_param;

	number_of_timesteps_per_device_spike_copy_check = number_of_timesteps_per_device_spike_copy_check_param;
	device_spike_store_size_multiple_of_total_neurons = device_spike_store_size_multiple_of_total_neurons_param;
	proportion_of_device_spike_store_full_before_copy = proportion_of_device_spike_store_full_before_copy_param;
	size_of_device_spike_store = device_spike_store_size_multiple_of_total_neurons * neurons->total_number_of_neurons;

	d_per_neuron_spike_counts = NULL;
	
	d_neuron_ids_of_stored_spikes_on_device = NULL;
	h_neuron_ids_of_stored_spikes_on_host = NULL;
	
	d_time_in_seconds_of_stored_spikes_on_device= NULL;
	h_time_in_seconds_of_stored_spikes_on_host = NULL;

	d_total_number_of_spikes_stored_on_device = NULL;
	h_total_number_of_spikes_stored_on_host = 0;

	reset_neuron_ids = NULL;
	reset_neuron_times = NULL;

}


// RecordingElectrodes Destructor
RecordingElectrodes::~RecordingElectrodes() {

	CudaSafeCall(cudaFree(d_total_number_of_spikes_stored_on_device));
	CudaSafeCall(cudaFree(d_neuron_ids_of_stored_spikes_on_device));
	CudaSafeCall(cudaFree(d_time_in_seconds_of_stored_spikes_on_device));

	free(h_neuron_ids_of_stored_spikes_on_host);
	free(h_time_in_seconds_of_stored_spikes_on_host);
	free(h_total_number_of_spikes_stored_on_device);

}


void RecordingElectrodes::allocate_pointers_for_spike_store() {

	h_total_number_of_spikes_stored_on_device = (int*)malloc(sizeof(int));
	
	CudaSafeCall(cudaMalloc((void **)&d_neuron_ids_of_stored_spikes_on_device, sizeof(int)*size_of_device_spike_store));
	CudaSafeCall(cudaMalloc((void **)&d_time_in_seconds_of_stored_spikes_on_device, sizeof(float)*size_of_device_spike_store));
	CudaSafeCall(cudaMalloc((void **)&d_total_number_of_spikes_stored_on_device, sizeof(int)));

	reset_neuron_ids = (int *)malloc(sizeof(int)*size_of_device_spike_store);
	reset_neuron_times = (float *)malloc(sizeof(float)*size_of_device_spike_store);
	for (int i=0; i < size_of_device_spike_store; i++){
		reset_neuron_ids[i] = -1;
		reset_neuron_times[i] = -1.0f;
	}
}


void RecordingElectrodes::reset_pointers_for_spike_store() {

	h_total_number_of_spikes_stored_on_device[0] = 0;
	h_total_number_of_spikes_stored_on_host = 0;

	CudaSafeCall(cudaMemset(d_neuron_ids_of_stored_spikes_on_device, -1, sizeof(int)*size_of_device_spike_store));
	CudaSafeCall(cudaMemset(d_time_in_seconds_of_stored_spikes_on_device, -1.0f, sizeof(float)*size_of_device_spike_store));
	CudaSafeCall(cudaMemset(d_total_number_of_spikes_stored_on_device, 0, sizeof(int)));

}



void RecordingElectrodes::allocate_pointers_for_spike_count() {
	//For counting spikes
	CudaSafeCall(cudaMalloc((void **)&d_per_neuron_spike_counts, sizeof(int) * neurons->total_number_of_neurons));
	
}

void RecordingElectrodes::reset_pointers_for_spike_count() {

	CudaSafeCall(cudaMemset(d_per_neuron_spike_counts, 0, sizeof(int) * neurons->total_number_of_neurons));

}




void RecordingElectrodes::collect_spikes_for_timestep(float current_time_in_seconds) {
	collect_spikes_for_timestep_kernel<<<neurons->number_of_neuron_blocks_per_grid, neurons->threads_per_block>>>(neurons->d_last_spike_time_of_each_neuron,
														d_total_number_of_spikes_stored_on_device,
														d_neuron_ids_of_stored_spikes_on_device,
														d_time_in_seconds_of_stored_spikes_on_device,
														current_time_in_seconds,
														neurons->total_number_of_neurons);

	CudaCheckError();
}

void RecordingElectrodes::copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch) {

	if (((timestep_index % number_of_timesteps_per_device_spike_copy_check) == 0) || (timestep_index == (number_of_timesteps_per_epoch-1))){

		// Finally, we want to get the spikes back. Every few timesteps check the number of spikes:
		CudaSafeCall(cudaMemcpy(&(h_total_number_of_spikes_stored_on_device[0]), &(d_total_number_of_spikes_stored_on_device[0]), (sizeof(int)), cudaMemcpyDeviceToHost));

		// Ensure that we don't have too many
		if (h_total_number_of_spikes_stored_on_device[0] > size_of_device_spike_store){
			print_message_and_exit("Spike recorder has been overloaded! Reduce threshold.");
		}

		// Deal with them!
		if ((h_total_number_of_spikes_stored_on_device[0] >= (proportion_of_device_spike_store_full_before_copy * size_of_device_spike_store)) ||  (timestep_index == (number_of_timesteps_per_epoch - 1))){

			// Reallocate host spike arrays to accommodate for new device spikes.
			h_neuron_ids_of_stored_spikes_on_host = (int*)realloc(h_neuron_ids_of_stored_spikes_on_host, sizeof(int)*(h_total_number_of_spikes_stored_on_host + h_total_number_of_spikes_stored_on_device[0]));
			h_time_in_seconds_of_stored_spikes_on_host = (float*)realloc(h_time_in_seconds_of_stored_spikes_on_host, sizeof(float)*(h_total_number_of_spikes_stored_on_host + h_total_number_of_spikes_stored_on_device[0]));

			// Copy device spikes into correct host array location
			CudaSafeCall(cudaMemcpy((void*)&h_neuron_ids_of_stored_spikes_on_host[h_total_number_of_spikes_stored_on_host], 
									d_neuron_ids_of_stored_spikes_on_device, 
									(sizeof(int)*h_total_number_of_spikes_stored_on_device[0]), 
									cudaMemcpyDeviceToHost));
			CudaSafeCall(cudaMemcpy((void*)&h_time_in_seconds_of_stored_spikes_on_host[h_total_number_of_spikes_stored_on_host], 
									d_time_in_seconds_of_stored_spikes_on_device, 
									sizeof(float)*h_total_number_of_spikes_stored_on_device[0], 
									cudaMemcpyDeviceToHost));

			h_total_number_of_spikes_stored_on_host += h_total_number_of_spikes_stored_on_device[0];


			// Reset device spikes
			CudaSafeCall(cudaMemset(&(d_total_number_of_spikes_stored_on_device[0]), 0, sizeof(int)));
			CudaSafeCall(cudaMemcpy(d_neuron_ids_of_stored_spikes_on_device, reset_neuron_ids, sizeof(int)*size_of_device_spike_store, cudaMemcpyHostToDevice));
			CudaSafeCall(cudaMemcpy(d_time_in_seconds_of_stored_spikes_on_device, reset_neuron_times, sizeof(float)*size_of_device_spike_store, cudaMemcpyHostToDevice));
			h_total_number_of_spikes_stored_on_device[0] = 0;
		}
	}
}


void RecordingElectrodes::add_spikes_to_per_neuron_spike_count(float current_time_in_seconds) {
	add_spikes_to_per_neuron_spike_count_kernel<<<neurons->number_of_neuron_blocks_per_grid, neurons->threads_per_block>>>(neurons->d_last_spike_time_of_each_neuron,
														d_per_neuron_spike_counts,
														current_time_in_seconds,
														neurons->total_number_of_neurons);
	CudaCheckError();
}




void RecordingElectrodes::write_spikes_to_file(int epoch_number, bool human_readable_storage, bool isTrained) {

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

	if (human_readable_storage){
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



void RecordingElectrodes::delete_and_reset_recorded_spikes() {

	// Reset the spike store
	// Host values
	h_total_number_of_spikes_stored_on_host = 0;
	h_total_number_of_spikes_stored_on_device[0] = 0;
	// Free/Clear Device stuff
	// Reset the number on the device
	CudaSafeCall(cudaMemset(&(d_total_number_of_spikes_stored_on_device[0]), 0, sizeof(int)));
	CudaSafeCall(cudaMemset(d_neuron_ids_of_stored_spikes_on_device, -1, sizeof(int)*neurons->total_number_of_neurons));
	CudaSafeCall(cudaMemset(d_time_in_seconds_of_stored_spikes_on_device, -1.0f, sizeof(float)*neurons->total_number_of_neurons));

	// Free malloced host stuff
	free(h_neuron_ids_of_stored_spikes_on_host);
	free(h_time_in_seconds_of_stored_spikes_on_host);
	h_neuron_ids_of_stored_spikes_on_host = NULL;
	h_time_in_seconds_of_stored_spikes_on_host = NULL;
}



__global__ void add_spikes_to_per_neuron_spike_count_kernel(float* d_last_spike_time_of_each_neuron,
								int* d_per_neuron_spike_counts,
								float current_time_in_seconds,
								size_t total_number_of_neurons) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while (idx < total_number_of_neurons) {

		if (d_last_spike_time_of_each_neuron[idx] == current_time_in_seconds) {
			atomicAdd(&d_per_neuron_spike_counts[idx], 1);
		}

		// if (idx == 1000) printf("d_per_neuron_spike_counts[idx]: %d\n", d_per_neuron_spike_counts[idx]);
		idx += blockDim.x * gridDim.x;
	}
}

// Collect Spikes
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



void RecordingElectrodes::write_initial_synaptic_weights_to_file(SpikingSynapses *synapses, bool human_readable_storage) {
	ofstream initweightfile;
	if (human_readable_storage){
		initweightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights_Initial.txt", ios::out | ios::binary);
		for (int i=0; i < synapses->total_number_of_synapses; i++){
			initweightfile << to_string(synapses->synaptic_efficacies_or_weights[i]) << endl;

		}
		initweightfile.close();
	} else {
		initweightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights_Initial.bin", ios::out | ios::binary);
		initweightfile.write((char *)synapses->synaptic_efficacies_or_weights, synapses->total_number_of_synapses*sizeof(float));
		initweightfile.close();
	}
}


void RecordingElectrodes::write_network_state_to_file(SpikingSynapses *synapses, bool human_readable_storage) {

	clock_t save_network_state_start = clock();

	// Copy back the data that we might want:
	CudaSafeCall(cudaMemcpy(synapses->synaptic_efficacies_or_weights, synapses->d_synaptic_efficacies_or_weights, sizeof(float)*synapses->total_number_of_synapses, cudaMemcpyDeviceToHost));
	
	if (human_readable_storage){
		// Creating and Opening all the files
		ofstream synapsepre, synapsepost, weightfile, delayfile;
		weightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights.txt", ios::out | ios::binary);
		delayfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkDelays.txt", ios::out | ios::binary);
		synapsepre.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPre.txt", ios::out | ios::binary);
		synapsepost.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPost.txt", ios::out | ios::binary);
		
		// Writing the data
		for (int i=0; i < synapses->total_number_of_synapses; i++){
			weightfile << to_string(synapses->synaptic_efficacies_or_weights[i]) << endl;
			delayfile << to_string(synapses->delays[i]) << endl;
			synapsepre << to_string(synapses->presynaptic_neuron_indices[i]) << endl;
			synapsepost << to_string(synapses->postsynaptic_neuron_indices[i]) << endl;
		}

		// Close files
		weightfile.close();
		delayfile.close();
		synapsepre.close();
		synapsepost.close();
	} else {
		// Creating and Opening all the files
		ofstream synapsepre, synapsepost, weightfile, delayfile;
		weightfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkWeights.bin", ios::out | ios::binary);
		delayfile.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkDelays.bin", ios::out | ios::binary);
		synapsepre.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPre.bin", ios::out | ios::binary);
		synapsepost.open(full_directory_name_for_simulation_data_files + prefix_string + "_NetworkPost.bin", ios::out | ios::binary);
		
		// Writing the data
		weightfile.write((char *)synapses->synaptic_efficacies_or_weights, synapses->total_number_of_synapses*sizeof(float));
		delayfile.write((char *)synapses->delays, synapses->total_number_of_synapses*sizeof(int));
		synapsepre.write((char *)synapses->presynaptic_neuron_indices, synapses->total_number_of_synapses*sizeof(int));
		synapsepost.write((char *)synapses->postsynaptic_neuron_indices, synapses->total_number_of_synapses*sizeof(int));

		// Close files
		weightfile.close();
		delayfile.close();
		synapsepre.close();
		synapsepost.close();
	}

	#ifndef QUIETSTART
	clock_t save_network_state_end = clock();
	float save_network_state_total_time = float(save_network_state_end - save_network_state_start) / CLOCKS_PER_SEC;
	printf("Network state saved to file.\n Time taken: %f\n", save_network_state_total_time);
	print_line_of_dashes_with_blank_lines_either_side();
	#endif

}


