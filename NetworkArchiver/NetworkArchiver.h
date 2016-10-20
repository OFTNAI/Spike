#ifndef NetworkArchiver_H
#define NetworkArchiver_H

#include <cuda.h>
#include <string>

#include "../Neurons/SpikingNeurons.h"
#include "../Synapses/SpikingSynapses.h"

class NetworkArchiver {
public:

	// Variables
	std::string full_directory_name_for_simulation_data_files;
	const char * prefix_string;

	// Host Pointers
	SpikingNeurons * neurons;
	SpikingSynapses * synapses;

	// Device Pointers


	// Constructor/Destructor
	NetworkArchiver(SpikingNeurons * neurons_parameter, SpikingSynapses * spiking_synapses, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
	~NetworkArchiver();

	void write_initial_synaptic_weights_to_file(SpikingSynapses *synapses, bool human_readable_storage);
	void write_network_state_to_file(SpikingSynapses *synapses, bool human_readable_storage);


private:

};

#endif
