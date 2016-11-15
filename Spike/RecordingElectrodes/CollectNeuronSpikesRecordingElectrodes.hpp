#ifndef CollectNeuronSpikesRecordingElectrodes_H
#define CollectNeuronSpikesRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.hpp"


struct Collect_Neuron_Spikes_Optional_Parameters {

	Collect_Neuron_Spikes_Optional_Parameters(): number_of_timesteps_per_device_spike_copy_check(50), device_spike_store_size_multiple_of_total_neurons(52), proportion_of_device_spike_store_full_before_copy(0.2), human_readable_storage(false) {}

	int number_of_timesteps_per_device_spike_copy_check;
	int device_spike_store_size_multiple_of_total_neurons;
	float proportion_of_device_spike_store_full_before_copy;
	bool human_readable_storage;

};




class CollectNeuronSpikesRecordingElectrodes : public RecordingElectrodes {
public:

	// Variables
	int size_of_device_spike_store;
	int h_total_number_of_spikes_stored_on_host;

	// Host Pointers
	Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters;
	int* h_neuron_ids_of_stored_spikes_on_host;
	int* h_total_number_of_spikes_stored_on_device;
	float* h_time_in_seconds_of_stored_spikes_on_host;

	// Device Pointers
	int* d_neuron_ids_of_stored_spikes_on_device;
	int* d_total_number_of_spikes_stored_on_device;
	float* d_time_in_seconds_of_stored_spikes_on_device;


	// Constructor/Destructor
	CollectNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
	~CollectNeuronSpikesRecordingElectrodes();

	void initialise_collect_neuron_spikes_recording_electrodes(Collect_Neuron_Spikes_Optional_Parameters * collect_neuron_spikes_optional_parameters_param);
	void allocate_pointers_for_spike_store();

	void collect_spikes_for_timestep(float current_time_in_seconds);
	void copy_spikes_from_device_to_host_and_reset_device_spikes_if_device_spike_count_above_threshold(float current_time_in_seconds, int timestep_index, int number_of_timesteps_per_epoch);
	void write_spikes_to_file(int epoch_number, bool isTrained);

	void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);

	void delete_and_reset_collected_spikes();

private:

	// Host Pointers
	int* reset_neuron_ids;
	float* reset_neuron_times;

};

/*CUDA
__global__ void collect_spikes_for_timestep_kernel(float* d_last_spike_time_of_each_neuron,
								int* d_total_number_of_spikes_stored_on_device,
								int* d_neuron_ids_of_stored_spikes_on_device,
								float* d_time_in_seconds_of_stored_spikes_on_device,
								float current_time_in_seconds,
								size_t total_number_of_neurons);
*/


#endif
