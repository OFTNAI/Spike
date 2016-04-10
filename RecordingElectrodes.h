//	RecordingElectrodes Class header
//	RecordingElectrodes.h
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#ifndef RecordingElectrodes_H
#define RecordingElectrodes_H

#include <cuda.h>

#include "Neurons.h"
#include "Connections.h"

class RecordingElectrodes{
public:

	int* d_tempstorenum;
	int* d_tempstoreID;
	float* d_tempstoretimes;

	int* h_tempstoreID;
	float* h_tempstoretimes;
	int* h_temp_total_number_of_spikes;

	int h_total_number_of_spikes;
	int* h_spikestoreID;
	float* h_spikestoretimes;


	// Constructor/Destructor
	RecordingElectrodes();
	~RecordingElectrodes();

	void initialise_device_pointers(int total_number_of_neurons);
	void initialise_host_pointers(int total_number_of_neurons);

	void save_spikes_to_host(Neurons *neurons, float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch, dim3 number_of_neuron_blocks_per_grid, dim3 threads_per_block);
	void write_spikes_to_file(Neurons *neurons, int epoch_number);

	void write_initial_synaptic_weights_to_file(Connections *connections);

};

__global__ void spikeCollect(float* d_lastspiketime,
								int* d_tempstorenum,
								int* d_tempstoreID,
								float* d_tempstoretimes,
								float current_time_in_seconds,
								size_t total_number_of_neurons);



#endif