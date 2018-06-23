#include "SpikingNeurons.hpp"
#include <stdlib.h>

// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {
}

// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {
  free(after_spike_reset_potentials_vreset);
  free(thresholds_for_action_potential_spikes);
  free(bitarray_of_neuron_spikes);
}

void SpikingNeurons::prepare_backend_early() {
}

int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
  int new_group_id = Neurons::AddGroup(group_params);

  spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;

  after_spike_reset_potentials_vreset = (float*)realloc(after_spike_reset_potentials_vreset, (total_number_of_neurons*sizeof(float)));
  resting_potentials_v0 = (float*)realloc(resting_potentials_v0, (total_number_of_neurons*sizeof(float)));
  thresholds_for_action_potential_spikes = (float*)realloc(thresholds_for_action_potential_spikes, (total_number_of_neurons*sizeof(float)));

  for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
    after_spike_reset_potentials_vreset[i] = spiking_group_params->after_spike_reset_potential_vreset;
    resting_potentials_v0[i] = spiking_group_params->resting_potential_v0;
    thresholds_for_action_potential_spikes[i] = spiking_group_params->threshold_for_action_potential_spike;
  }

  return new_group_id;
}

void SpikingNeurons::state_update(float current_time_in_seconds, float timestep) {
  backend()->state_update(current_time_in_seconds, timestep);
}

SPIKE_MAKE_INIT_BACKEND(SpikingNeurons);
