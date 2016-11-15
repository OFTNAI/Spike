#ifndef RecordingElectrodes_H
#define RecordingElectrodes_H

//CUDA #include <cuda.h>
#include <string>
using namespace std;

#include "../Neurons/SpikingNeurons.hpp"
#include "../Synapses/SpikingSynapses.hpp"

class RecordingElectrodes {
public:

	// Variables
	std::string full_directory_name_for_simulation_data_files;
	const char * prefix_string;

	// Host Pointers
	SpikingNeurons * neurons;
	SpikingSynapses * synapses;

	// Device Pointers


	// Constructor/Destructor
	RecordingElectrodes();
	RecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param);
	~RecordingElectrodes();



private:

	// Host Pointers

};





#endif
