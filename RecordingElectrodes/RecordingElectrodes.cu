#include "RecordingElectrodes.h"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/CUDAErrorCheckHelpers.h"
#include "../Helpers/TerminalHelpers.h"
#include <string>
#include <time.h>
using namespace std;

// RecordingElectrodes Constructor
RecordingElectrodes::RecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * spiking_synapses, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param) {

	// Variables
	full_directory_name_for_simulation_data_files = full_directory_name_for_simulation_data_files_param;
	prefix_string = prefix_string_param;

	// Host Pointers
	neurons = neurons_parameter;
	synapses = synapses_parameter;

}


// RecordingElectrodes Destructor
RecordingElectrodes::~RecordingElectrodes() {


}

