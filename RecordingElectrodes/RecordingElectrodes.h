//	RecordingElectrodes Class header
//	RecordingElectrodes.h
//
//  Adapted from CUDACode
//	Authors: Nasir Ahmad and James Isbister
//	Date: 9/4/2016

#ifndef RecordingElectrodes_H
#define RecordingElectrodes_H

#include <cuda.h>
#include <string>

#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/SpikingSynapses.h"

class RecordingElectrodes{
public:

	int * d_per_neuron_spike_counts;

	int number_of_timesteps_per_device_spike_copy_check;
	int device_spike_store_size_multiple_of_total_neurons;
	float proportion_of_device_spike_store_full_before_copy;

	int size_of_device_spike_store;

	std::string RESULTS_DIRECTORY;

	int* d_neuron_ids_of_stored_spikes_on_device;
	int* h_neuron_ids_of_stored_spikes_on_device;
	int* h_neuron_ids_of_stored_spikes_on_host;

	float* d_time_in_seconds_of_stored_spikes_on_device;
	float* h_time_in_seconds_of_stored_spikes_on_device;
	float* h_time_in_seconds_of_stored_spikes_on_host;

	// Various spikes stored totals
	int* d_total_number_of_spikes_stored_on_device;
	int* h_total_number_of_spikes_stored_on_device;
	int h_total_number_of_spikes_stored_on_host;

	const char * prefix_string;

	SpikingNeurons * neurons;

	// Constructor/Destructor
	RecordingElectrodes(SpikingNeurons * neurons_parameter, std::string RESULTS_DIRECTORY_param, const char * prefix_string_param, int number_of_timesteps_per_device_spike_copy_check_param, int device_spike_store_size_multiple_of_total_neurons_param, float proportion_of_device_spike_store_full_before_copy_param);
	~RecordingElectrodes();

	void allocate_pointers_for_spike_store();
	void reset_pointers_for_spike_store();
	void allocate_pointers_for_spike_count();
	void reset_pointers_for_spike_count();


	void collect_spikes_for_timestep(float current_time_in_seconds);
	void copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch);
	void write_spikes_to_file(int epoch_number, bool human_readable_storage, bool isTrained);

	void write_initial_synaptic_weights_to_file(SpikingSynapses *synapses, bool human_readable_storage);
	void write_network_state_to_file(SpikingSynapses *synapses, bool human_readable_storage);

	void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

	void delete_and_reset_recorded_spikes();

};

__global__ void add_spikes_to_per_neuron_spike_count_kernel(float* d_last_spike_time_of_each_neuron,
								int* d_per_neuron_spike_counts,
								float current_time_in_seconds,
								size_t total_number_of_neurons);

__global__ void collect_spikes_for_timestep_kernel(float* d_last_spike_time_of_each_neuron,
								int* d_total_number_of_spikes_stored_on_device,
								int* d_neuron_ids_of_stored_spikes_on_device,
								float* d_time_in_seconds_of_stored_spikes_on_device,
								float current_time_in_seconds,
								size_t total_number_of_neurons);



#endif
