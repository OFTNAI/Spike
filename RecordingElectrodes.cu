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

void RecordingElectrodes::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	threads_per_block.x = threads;

}




