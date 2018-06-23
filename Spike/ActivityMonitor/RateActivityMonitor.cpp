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

void RateActivityMonitor::reset_state() {
  for (int n = 0; n < neurons->total_number_of_neurons; n++){
    per_neuron_spike_counts[n] = 0;
  }
  backend()->reset_state();
}

void RateActivityMonitor::prepare_backend_early() {
  per_neuron_spike_counts = (int *)malloc(sizeof(int)*neurons->total_number_of_neurons);
}

void RateActivityMonitor::state_update(float current_time_in_seconds, float timestep) {
  backend()->add_spikes_to_per_neuron_spike_count(current_time_in_seconds);
}

void RateActivityMonitor::final_update(float current_time_in_seconds, float timestep){
  backend()->copy_spike_count_to_host();
}

SPIKE_MAKE_INIT_BACKEND(RateActivityMonitor);
