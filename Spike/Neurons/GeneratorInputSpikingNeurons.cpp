#include "GeneratorInputSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>

GeneratorInputSpikingNeurons::~GeneratorInputSpikingNeurons() {
  free(neuron_id_matrix_for_stimuli);
  free(spike_times_matrix_for_stimuli);
  free(number_of_spikes_in_stimuli);
  free(temporal_lengths_of_stimuli);
}

/* Don't need this as it is inherited without change:
// Add Group of given size as usual - nothing special in constructor
int GeneratorInputSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){
	
	int new_group_id = InputSpikingNeurons::AddGroup(group_params);
	return new_group_id;

}
*/

void GeneratorInputSpikingNeurons::state_update(float current_time_in_seconds,  float timestep){
  backend()->state_update(current_time_in_seconds, timestep);
}

void GeneratorInputSpikingNeurons::AddStimulus(int spikenumber, int* ids, float* spiketimes){
  ++total_number_of_input_stimuli;
  // If the number of spikes in this stimulus is larger than any other ...
  if (spikenumber > length_of_longest_stimulus){
    length_of_longest_stimulus = spikenumber;
  }

  number_of_spikes_in_stimuli = (int*)realloc(number_of_spikes_in_stimuli, sizeof(int)*total_number_of_input_stimuli);
  temporal_lengths_of_stimuli = (float*)realloc(temporal_lengths_of_stimuli, sizeof(float)*total_number_of_input_stimuli);
  neuron_id_matrix_for_stimuli = (int**)realloc(neuron_id_matrix_for_stimuli, sizeof(int*)*total_number_of_input_stimuli);
  spike_times_matrix_for_stimuli = (float**)realloc(spike_times_matrix_for_stimuli, sizeof(float*)*total_number_of_input_stimuli);
	
  // Initialize matrices
  neuron_id_matrix_for_stimuli[total_number_of_input_stimuli - 1] = nullptr;
  spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1] = nullptr;
  number_of_spikes_in_stimuli[total_number_of_input_stimuli - 1] = 0;
	
  neuron_id_matrix_for_stimuli[total_number_of_input_stimuli - 1] = (int*)
    realloc(neuron_id_matrix_for_stimuli[total_number_of_input_stimuli - 1], 
            sizeof(int)*(spikenumber));
  spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1] = (float*)
    realloc(spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1], 
            sizeof(float)*(spikenumber));
	
  // Assign the genid values according to how many neurons exist already
  for (int i = 0; i < spikenumber; i++){
    neuron_id_matrix_for_stimuli[total_number_of_input_stimuli - 1][i] = ids[i];
    spike_times_matrix_for_stimuli[total_number_of_input_stimuli - 1][i] = spiketimes[i];
  }

  // Increment the number of entries the generator population
  number_of_spikes_in_stimuli[total_number_of_input_stimuli - 1] = spikenumber;

  // Get the maximum length of this stimulus
  float maxlen = 0.0f;
  for (int spikeidx = 0; spikeidx < spikenumber; spikeidx++){
    if (spiketimes[spikeidx] > maxlen){
      maxlen = spiketimes[spikeidx];
    }
  }
  // Set the correct array location for maximum length
  temporal_lengths_of_stimuli[total_number_of_input_stimuli - 1] = maxlen;
}

SPIKE_MAKE_INIT_BACKEND(GeneratorInputSpikingNeurons);
