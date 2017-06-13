#include "PoissonInputSpikingNeurons.hpp"
#include "Spike/Helpers/TerminalHelpers.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <algorithm> // For random shuffle
#include <cassert>

using namespace std;

PoissonInputSpikingNeurons::~PoissonInputSpikingNeurons() {
  free(random_state_manager);
  free(rates);
}


int PoissonInputSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
  int new_group_id = InputSpikingNeurons::AddGroup(group_params);
  poisson_input_spiking_neuron_parameters_struct * poisson_input_spiking_group_params = (poisson_input_spiking_neuron_parameters_struct*)group_params;
  rate = poisson_input_spiking_group_params->rate;
  return new_group_id;
}


void PoissonInputSpikingNeurons::set_up_rates() {
  rates = (float*)realloc(rates, sizeof(float)*total_number_of_neurons);
  for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
    rates[i] = rate;
  }
  total_number_of_transformations_per_object = 1;
  total_number_of_objects = 1;
  total_number_of_input_stimuli = 1;
}


void PoissonInputSpikingNeurons::init_random_state(bool force) {
  assert(backend() && "Backend needs to have been prepared before calling this!");
  if (force || !random_state_manager) {
    random_state_manager = new RandomStateManager();
    random_state_manager->init_backend(backend()->context);
  }
}

void PoissonInputSpikingNeurons::prepare_backend_early() {
  InputSpikingNeurons::prepare_backend_early();
  init_random_state();
}

void PoissonInputSpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
  backend()->state_update(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(PoissonInputSpikingNeurons);
