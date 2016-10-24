#ifndef NetworkStateArchiveRecordingElectrodes_H
#define NetworkStateArchiveRecordingElectrodes_H


#include "../RecordingElectrodes/RecordingElectrodes.h"


struct Network_State_Archive_Optional_Parameters {

	Network_State_Archive_Optional_Parameters(): human_readable_storage(false) {}

		bool human_readable_storage;
	
};


class NetworkStateArchiveRecordingElectrodes  : public RecordingElectrodes {
public:

	// Variables

	// Host Pointers
	Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters;

	// Device Pointers


	// Constructor/Destructor
	NetworkStateArchiveRecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * spiking_synapses, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
	~NetworkStateArchiveRecordingElectrodes();

	void initialise_network_state_archive_recording_electrodes(Network_State_Archive_Optional_Parameters * network_state_archive_optional_parameters_param);

	void write_initial_synaptic_weights_to_file();
	void write_network_state_to_file();


private:

};

#endif
