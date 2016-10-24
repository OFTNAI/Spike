#include "CountNeuronSpikesRecordingElectrodes.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"
#include <string>
#include <time.h>
using namespace std;

// CountNeuronSpikesRecordingElectrodes Constructor
CountNeuronSpikesRecordingElectrodes::CountNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param) {

	// Variables

	// Host Pointers

	// Device Pointers
	d_per_neuron_spike_counts = NULL;

	// Private Host Pointeres


}


// CountNeuronSpikesRecordingElectrodes Destructor
CountNeuronSpikesRecordingElectrodes::~CountNeuronSpikesRecordingElectrodes() {

	CudaSafeCall(cudaFree(d_per_neuron_spike_counts));

}


void CountNeuronSpikesRecordingElectrodes::initialise_count_neuron_spikes_recording_electrodes() {

	allocate_pointers_for_spike_count();
	reset_pointers_for_spike_count();

}


void CountNeuronSpikesRecordingElectrodes::allocate_pointers_for_spike_count() {
	//For counting spikes
	CudaSafeCall(cudaMalloc((void **)&d_per_neuron_spike_counts, sizeof(int) * neurons->total_number_of_neurons));
	
}

void CountNeuronSpikesRecordingElectrodes::reset_pointers_for_spike_count() {

	CudaSafeCall(cudaMemset(d_per_neuron_spike_counts, 0, sizeof(int) * neurons->total_number_of_neurons));

}


void CountNeuronSpikesRecordingElectrodes::add_spikes_to_per_neuron_spike_count(float current_time_in_seconds) {
	add_spikes_to_per_neuron_spike_count_kernel<<<neurons->number_of_neuron_blocks_per_grid, neurons->threads_per_block>>>(neurons->d_last_spike_time_of_each_neuron,
														d_per_neuron_spike_counts,
														current_time_in_seconds,
														neurons->total_number_of_neurons);
	CudaCheckError();
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