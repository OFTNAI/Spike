#ifndef CollectNeuronSpikesRecordingElectrodes_H
#define CollectNeuronSpikesRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.h"


struct Collect_Neuron_Spikes_Optional_Parameters {

	Collect_Neuron_Spikes_Optional_Parameters(): number_of_timesteps_per_device_spike_copy_check_param(50), device_spike_store_size_multiple_of_total_neurons_param(52), proportion_of_device_spike_store_full_before_copy_param(0.2) {}

	int number_of_timesteps_per_device_spike_copy_check_param;
	int device_spike_store_size_multiple_of_total_neurons_param;
	float proportion_of_device_spike_store_full_before_copy_param;

};




class CollectNeuronSpikesRecordingElectrodes : public RecordingElectrodes {
public:

	// Variables
	int number_of_timesteps_per_device_spike_copy_check;
	int device_spike_store_size_multiple_of_total_neurons;
	int size_of_device_spike_store;
	int h_total_number_of_spikes_stored_on_host;
	float proportion_of_device_spike_store_full_before_copy;

	// Host Pointers
	int* h_neuron_ids_of_stored_spikes_on_host;
	int* h_total_number_of_spikes_stored_on_device;
	float* h_time_in_seconds_of_stored_spikes_on_host;

	// Device Pointers
	int* d_neuron_ids_of_stored_spikes_on_device;
	int* d_total_number_of_spikes_stored_on_device;
	float* d_time_in_seconds_of_stored_spikes_on_device;


	// Constructor/Destructor
	CollectNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * spiking_synapses, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
	~CollectNeuronSpikesRecordingElectrodes();

	void initialise_collect_neuron_spikes_recording_electrodes(Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters);
	void allocate_pointers_for_spike_count();
	void reset_pointers_for_spike_count();

	void collect_spikes_for_timestep(float current_time_in_seconds);
	void copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch);
	void write_spikes_to_file(int epoch_number, bool human_readable_storage, bool isTrained);

	void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

	void delete_and_reset_collected_spikes();

private:

	// Host Pointers
	int* reset_neuron_ids;
	float* reset_neuron_times;

};

__global__ void collect_spikes_for_timestep_kernel(float* d_last_spike_time_of_each_neuron,
								int* d_total_number_of_spikes_stored_on_device,
								int* d_neuron_ids_of_stored_spikes_on_device,
								float* d_time_in_seconds_of_stored_spikes_on_device,
								float current_time_in_seconds,
								size_t total_number_of_neurons);



#endif
