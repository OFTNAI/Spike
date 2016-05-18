//	RecordingElectrodes Class header
//	RecordingElectrodes.h
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#ifndef RecordingElectrodes_H
#define RecordingElectrodes_H

#include <cuda.h>

#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/SpikingSynapses.h"

class RecordingElectrodes{
public:

	int* d_tempstorenum;
	int* d_tempstoreID;
	float* d_tempstoretimes;

	int* h_tempstoreID;
	float* h_tempstoretimes;
	int* h_temp_total_number_of_spikes;
	int* h_spikestoreID;
	float* h_spikestoretimes;
	int h_total_number_of_spikes;

	const char * prefix_string;

	SpikingNeurons * neurons;

	// Constructor/Destructor
	RecordingElectrodes(SpikingNeurons * neurons_parameter, const char * prefix_string_param);
	~RecordingElectrodes();

	void initialise_device_pointers();
	void initialise_host_pointers();

	void save_spikes_to_host(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch);
	void write_spikes_to_file(Neurons *neurons, int epoch_number);

	void write_initial_synaptic_weights_to_file(SpikingSynapses *synapses);
	void save_network_state(SpikingSynapses *synapses);

};

__global__ void spikeCollect(float* d_last_spike_time_of_each_neuron,
								int* d_tempstorenum,
								int* d_tempstoreID,
								float* d_tempstoretimes,
								float current_time_in_seconds,
								size_t total_number_of_neurons);



#endif