#include "Monitors.hpp"
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include "../Helpers/TerminalHelpers.hpp"
#include <time.h>

Monitors::Monitors(SpikingNeurons* neuron_set) {
  neurons = neuron_set;
}

void Monitors::reset_state() {
  backend()->reset_state();
}

SPIKE_MAKE_STUB_INIT_BACKEND(Monitors);
