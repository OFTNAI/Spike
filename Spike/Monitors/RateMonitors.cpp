#include "RateMonitors.hpp"
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// RateMonitors Constructor
RateMonitors::RateMonitors(SpikingNeurons * neurons_parameter) : Monitors(neurons_parameter) {
}

void RateMonitors::initialise_count_neuron_spikes_recording_electrodes() {
  reset_state();
  prepare_backend();
}

void RateMonitors::state_update(float current_time_in_seconds, float timestep) {
  backend()->add_spikes_to_per_neuron_spike_count(current_time_in_seconds);
}

SPIKE_MAKE_INIT_BACKEND(RateMonitors);
