#include "RecordingElectrodes.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
//CUDA #include "../Helpers/CUDAErrorCheckHelpers.hpp"
#include "../Helpers/TerminalHelpers.hpp"
#include <time.h>


// RecordingElectrodes Constructor

RecordingElectrodes::RecordingElectrodes() {
	
}

RecordingElectrodes::RecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param) {

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

