#ifndef NetworkStateArchiveRecordingElectrodes_H
#define NetworkStateArchiveRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.h"


class NetworkStateArchiveRecordingElectrodes  : public RecordingElectrodes {
public:

	// Variables
	std::string full_directory_name_for_simulation_data_files;
	const char * prefix_string;

	// Host Pointers
	SpikingNeurons * neurons;
	SpikingSynapses * synapses;

	// Device Pointers


	// Constructor/Destructor
	NetworkStateArchiveRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * spiking_synapses, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
	~NetworkStateArchiveRecordingElectrodes();

	void write_initial_synaptic_weights_to_file(SpikingSynapses *synapses, bool human_readable_storage);
	void write_network_state_to_file(SpikingSynapses *synapses, bool human_readable_storage);


private:

};

#endif
