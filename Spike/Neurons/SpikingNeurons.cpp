#include "SpikingNeurons.hpp"
#include <stdlib.h>

// SpikingNeurons Constructor
SpikingNeurons::SpikingNeurons() {
	// Variables
	bitarray_length = 0;
	bitarray_maximum_axonal_delay_in_timesteps = 0;
	high_fidelity_spike_flag = false;

	// Host Pointers
	after_spike_reset_membrane_potentials_c = nullptr;
	thresholds_for_action_potential_spikes = nullptr;
	bitarray_of_neuron_spikes = nullptr;
}

// SpikingNeurons Destructor
SpikingNeurons::~SpikingNeurons() {
	free(after_spike_reset_membrane_potentials_c);
	free(thresholds_for_action_potential_spikes);
	free(bitarray_of_neuron_spikes);
}

void SpikingNeurons::prepare_backend_early() {
  // Choosing Spike Mechanism
  high_fidelity_spike_flag = backend()->context->params.high_fidelity_spike_storage;
  bitarray_maximum_axonal_delay_in_timesteps = backend()->context->params.maximum_axonal_delay_in_timesteps;

  if (high_fidelity_spike_flag){
    // Create bit array of correct length
    bitarray_length = (bitarray_maximum_axonal_delay_in_timesteps / 8) + 1; // each char is 8 bit long.
    bitarray_of_neuron_spikes = (unsigned char *)malloc(sizeof(unsigned char)*bitarray_length*total_number_of_neurons);
    for (int i = 0; i < bitarray_length*total_number_of_neurons; i++){
      bitarray_of_neuron_spikes[i] = (unsigned char)0;
    }
  }
}

int SpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
	int new_group_id = Neurons::AddGroup(group_params);

	spiking_neuron_parameters_struct * spiking_group_params = (spiking_neuron_parameters_struct*)group_params;

	after_spike_reset_membrane_potentials_c = (float*)realloc(after_spike_reset_membrane_potentials_c, (total_number_of_neurons*sizeof(float)));
	thresholds_for_action_potential_spikes = (float*)realloc(thresholds_for_action_potential_spikes, (total_number_of_neurons*sizeof(float)));

	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		after_spike_reset_membrane_potentials_c[i] = spiking_group_params->resting_potential_v0;
		thresholds_for_action_potential_spikes[i] = spiking_group_params->threshold_for_action_potential_spike;
	}

	return new_group_id;
}

void SpikingNeurons::check_for_neuron_spikes(float current_time_in_seconds, float timestep) {
  backend()->check_for_neuron_spikes(current_time_in_seconds, timestep);
}

void SpikingNeurons::update_membrane_potentials(float timestep, float current_time_in_seconds) {
  backend()->update_membrane_potentials(timestep, current_time_in_seconds);
}

SPIKE_MAKE_STUB_INIT_BACKEND(SpikingNeurons);
