//	RecordingElectrodes Class C++
//	RecordingElectrodes.cu
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#include "RecordingElectrodes.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDAErrorCheckHelpers.h"


// RecordingElectrodes Constructor
RecordingElectrodes::RecordingElectrodes() {
	h_total_number_of_spikes = 0;
	h_spikestoreID = NULL;
	h_spikestoretimes = NULL;
}


// RecordingElectrodes Destructor
RecordingElectrodes::~RecordingElectrodes() {

}


void RecordingElectrodes::initialise_device_pointers(int total_number_of_neurons) {
	// For saving spikes (Make seperate class)
	CudaSafeCall(cudaMalloc((void **)&d_tempstoreID, sizeof(int)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_tempstoretimes, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMalloc((void **)&d_tempstorenum, sizeof(int)));

	// Send data to device: data for saving spikes
	CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*total_number_of_neurons));
	CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*total_number_of_neurons));
	CudaSafeCall(cudaMemset(d_tempstorenum, 0, sizeof(int)));
}

void RecordingElectrodes::initialise_host_pointers(int total_number_of_neurons) {
	h_tempstoreID = (int*)malloc(sizeof(int)*total_number_of_neurons);
	h_tempstoretimes = (float*)malloc(sizeof(float)*total_number_of_neurons);

	h_temp_total_number_of_spikes = (int*)malloc(sizeof(int));
	h_temp_total_number_of_spikes[0] = 0;
}

void RecordingElectrodes::save_spikes_to_host(Neurons *neurons, float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch, dim3 number_of_neuron_blocks_per_grid, dim3 threads_per_block) {

	// printf("Save spikes to host. total_number_of_neurons = %d.\n", neurons->total_number_of_neurons);

	// Storing the spikes that have occurred in this timestep
	spikeCollect<<<number_of_neuron_blocks_per_grid, threads_per_block>>>(neurons->d_lastspiketime,
														d_tempstorenum,
														d_tempstoreID,
														d_tempstoretimes,
														current_time_in_seconds,
														neurons->total_number_of_neurons);
	CudaCheckError();

	if (((timestep_index % 1) == 0) || (timestep_index == (number_of_timesteps_per_epoch-1))){

		// Finally, we want to get the spikes back. Every few timesteps check the number of spikes:
		CudaSafeCall(cudaMemcpy(&(h_temp_total_number_of_spikes[0]), &(d_tempstorenum[0]), (sizeof(int)), cudaMemcpyDeviceToHost));
		// Ensure that we don't have too many
		if (h_temp_total_number_of_spikes[0] > neurons->total_number_of_neurons){
			// ERROR!
			printf("Spike recorder has been overloaded! Reduce threshold. Exiting ...\n");
			exit(-1);
		}
		// Deal with them!
		if ((h_temp_total_number_of_spikes[0] >= (0.25*neurons->total_number_of_neurons)) ||  (timestep_index == (number_of_timesteps_per_epoch - 1))){

			// Allocate some memory for them:
			h_spikestoreID = (int*)realloc(h_spikestoreID, sizeof(int)*(h_total_number_of_spikes + h_temp_total_number_of_spikes[0]));
			h_spikestoretimes = (float*)realloc(h_spikestoretimes, sizeof(float)*(h_total_number_of_spikes + h_temp_total_number_of_spikes[0]));
			// Copy the data from device to host
			CudaSafeCall(cudaMemcpy(h_tempstoreID, 
									d_tempstoreID, 
									(sizeof(int)*(neurons->total_number_of_neurons)), 
									cudaMemcpyDeviceToHost));
			CudaSafeCall(cudaMemcpy(h_tempstoretimes, 
									d_tempstoretimes, 
									sizeof(float)*(neurons->total_number_of_neurons), 
									cudaMemcpyDeviceToHost));
			// Pop all of the times where they need to be:
			for (int l = 0; l < h_temp_total_number_of_spikes[0]; l++){
				h_spikestoreID[h_total_number_of_spikes + l] = h_tempstoreID[l];
				h_spikestoretimes[h_total_number_of_spikes + l] = h_tempstoretimes[l];
			}
			// Reset the number on the device
			CudaSafeCall(cudaMemset(&(d_tempstorenum[0]), 0, sizeof(int)));
			CudaSafeCall(cudaMemset(d_tempstoreID, -1, sizeof(int)*neurons->total_number_of_neurons));
			CudaSafeCall(cudaMemset(d_tempstoretimes, -1.0f, sizeof(float)*neurons->total_number_of_neurons));
			// Increase the number on host
			h_total_number_of_spikes += h_temp_total_number_of_spikes[0];
			h_temp_total_number_of_spikes[0] = 0;
		}
	}
}

// Collect Spikes
__global__ void spikeCollect(float* d_lastspiketime,
								int* d_tempstorenum,
								int* d_tempstoreID,
								float* d_tempstoretimes,
								float current_time_in_seconds,
								size_t total_number_of_neurons){

int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		// If a neuron has fired
		if (d_lastspiketime[idx] == current_time_in_seconds) {
			// Increase the number of spikes stored
			int i = atomicAdd(&d_tempstorenum[0], 1);
			// In the location, add the id and the time
			d_tempstoreID[i] = idx;
			d_tempstoretimes[i] = current_time_in_seconds;
		}
	}
	__syncthreads();
}

