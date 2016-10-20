#ifndef CountNeuronSpikesRecordingElectrodes_H
#define CountNeuronSpikesRecordingElectrodes_H

#include <cuda.h>
#include <string>

#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/SpikingSynapses.h"

class CountNeuronSpikesRecordingElectrodes : public RecordingElectrodes {
public:

	// Variables

	// Host Pointers

	// Device Pointers
	int * d_per_neuron_spike_counts;


	// Constructor/Destructor
	CountNeuronSpikesRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * spiking_synapses, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
	~CountNeuronSpikesRecordingElectrodes();

	void initialise_count_neuron_spikes_recording_electrodes();
	void allocate_pointers_for_spike_count();
	void reset_pointers_for_spike_count();

	void add_spikes_to_per_neuron_spike_count(float current_time_in_seconds);


private:

};

__global__ void add_spikes_to_per_neuron_spike_count_kernel(float* d_last_spike_time_of_each_neuron,
								int* d_per_neuron_spike_counts,
								float current_time_in_seconds,
								size_t total_number_of_neurons);


#endif
