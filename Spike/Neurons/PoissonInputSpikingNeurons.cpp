#include "PoissonInputSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>
#include "../Helpers/TerminalHelpers.hpp"
#include <algorithm> // For random shuffle
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


void PoissonInputSpikingNeurons::init_random_state() {
  random_state_manager = new RandomStateManager();
  random_state_manager->prepare_backend(backend()->context);
  printf("TODO: RNG should be managed globally...\n");
}

