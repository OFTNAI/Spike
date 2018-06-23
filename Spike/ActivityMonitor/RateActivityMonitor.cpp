#include "RateActivityMonitor.hpp"
#include <cassert>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <string>
#include <time.h>
using namespace std;

// RateActivityMonitor Constructor
RateActivityMonitor::RateActivityMonitor(SpikingNeurons * neurons_parameter){
  neurons = neurons_parameter;
}

void RateActivityMonitor::initialise_count_neuron_spikes_recording_electrodes() {
  reset_state();
  prepare_backend();
}

void RateActivityMonitor::state_update(float current_time_in_seconds, float timestep) {
  backend()->add_spikes_to_per_neuron_spike_count(current_time_in_seconds);
}

SPIKE_MAKE_INIT_BACKEND(RateActivityMonitor);
