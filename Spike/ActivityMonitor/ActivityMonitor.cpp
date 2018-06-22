#include "ActivityMonitor.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <time.h>

ActivityMonitor::Monitors(SpikingNeurons* neuron_set) {
  neurons = neuron_set;
}

void ActivityMonitor::reset_state() {
  backend()->reset_state();
}

SPIKE_MAKE_STUB_INIT_BACKEND(ActivityMonitor);
