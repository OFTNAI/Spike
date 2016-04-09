//	RecordingElectrodes Class header
//	RecordingElectrodes.h
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#ifndef RecordingElectrodes_H
#define RecordingElectrodes_H

#include <cuda.h>

#include "Connections.h"

class RecordingElectrodes{
public:

	int* d_tempstorenum;
	int* d_tempstoreID;
	float* d_tempstoretimes;

	int* h_tempstoreID;
	float* h_tempstoretimes;
	int* h_tempspikenum;

	int h_spikenum;
	int* h_spikestoreID;
	float* h_spikestoretimes;


	// Constructor/Destructor
	RecordingElectrodes();
	~RecordingElectrodes();

	void initialise_device_pointers(int total_number_of_neurons);
	void initialise_host_pointers(int total_number_of_neurons);

	void set_threads_per_block_and_blocks_per_grid(int threads);


private:
	dim3 number_of_neuron_blocks_per_grid;
	dim3 threads_per_block;

};



#endif