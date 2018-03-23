#include "RecordingElectrodes.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <time.h>

RecordingElectrodes::RecordingElectrodes(SpikingNeurons * neurons_parameter, SpikingSynapses * synapses_parameter, string full_directory_name_for_simulation_data_files_param, const char * prefix_string_param) {

	// Variables
	full_directory_name_for_simulation_data_files = full_directory_name_for_simulation_data_files_param;
	prefix_string = prefix_string_param;

	// Host Pointers
	neurons = neurons_parameter;
	synapses = synapses_parameter;

}

void RecordingElectrodes::reset_state() {
  backend()->reset_state();
}

SPIKE_MAKE_STUB_INIT_BACKEND(RecordingElectrodes);
