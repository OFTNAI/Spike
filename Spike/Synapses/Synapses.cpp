//	Synapse Class C++
//	Synapse.cpp
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#include "Synapses.hpp"
#include "../Helpers/TerminalHelpers.hpp"

#include <algorithm> // for random shuffle

// Synapses Constructor
Synapses::Synapses() {
  // On construction, seed
  // TODO: THE RANDOM SEED SHOULD BE A GLOBAL PARAMETER!!
  srand(42);	// Seeding the random numbers

  random_state_manager = new RandomStateManager();
}

void Synapses::prepare_backend_extra() {
  random_state_manager->prepare_backend(backend()->context);
}

// Synapses Destructor
Synapses::~Synapses() {
  free(presynaptic_neuron_indices);
  free(postsynaptic_neuron_indices);
  free(synaptic_efficacies_or_weights);
  free(original_synapse_indices);
  free(synapse_postsynaptic_neuron_count_index);
  free(random_state_manager);
  // free(number_of_synapse_blocks_per_grid);
}


void Synapses::AddGroup(int presynaptic_group_id, 
                        int postsynaptic_group_id, 
                        Neurons * neurons,
                        Neurons * input_neurons,
                        float timestep,
                        synapse_parameters_struct * synapse_params) {

	if (print_synapse_group_details == true) {
          printf("Adding synapse group...\n");
          printf("presynaptic_group_id: %d\n", presynaptic_group_id);
          printf("postsynaptic_group_id: %d\n", postsynaptic_group_id);
	}

	int* start_neuron_indices_for_neuron_groups = neurons->start_neuron_indices_for_each_group;
	int* start_neuron_indices_for_input_neuron_groups = input_neurons->start_neuron_indices_for_each_group;
	int* last_neuron_indices_for_neuron_groups = neurons->last_neuron_indices_for_each_group;
	int* last_neuron_indices_for_input_neuron_groups = input_neurons->last_neuron_indices_for_each_group;

	int * presynaptic_group_shape;
	int * postsynaptic_group_shape;

	int prestart = 0;
	int preend = 0;
	int poststart = 0;

	// Calculate presynaptic group start and end indices
	// Also assign presynaptic group shape
	bool presynaptic_group_is_input = PRESYNAPTIC_IS_INPUT(presynaptic_group_id);

	if (presynaptic_group_is_input) {

          // if (stdp_on == true) print_message_and_exit("Plasticity between input neurons and model neurons is not currently supported.");

          int corrected_presynaptic_group_id = CORRECTED_PRESYNAPTIC_ID(presynaptic_group_id, presynaptic_group_is_input);

          presynaptic_group_shape = input_neurons->group_shapes[corrected_presynaptic_group_id];

          if (presynaptic_group_id < -1){
            prestart = start_neuron_indices_for_input_neuron_groups[corrected_presynaptic_group_id];
          }
          preend = last_neuron_indices_for_input_neuron_groups[corrected_presynaptic_group_id] + 1;

	} else {

          presynaptic_group_shape = neurons->group_shapes[presynaptic_group_id];

          if (presynaptic_group_id > 0){
            prestart = start_neuron_indices_for_neuron_groups[presynaptic_group_id];
          }
          preend = last_neuron_indices_for_neuron_groups[presynaptic_group_id] + 1;

	}

	// Calculate postsynaptic group start and end indices
	// Also assign postsynaptic group shape
	if (postsynaptic_group_id < 0) { // If presynaptic group is Input group EXIT

          print_message_and_exit("Input groups cannot be a postsynaptic neuron group.");

	} else if (postsynaptic_group_id >= 0){
          postsynaptic_group_shape = neurons->group_shapes[postsynaptic_group_id];
          poststart = start_neuron_indices_for_neuron_groups[postsynaptic_group_id];
		
	}
	int postend = last_neuron_indices_for_neuron_groups[postsynaptic_group_id] + 1;

	if (print_synapse_group_details == true) {
          const char * presynaptic_group_type_string = (presynaptic_group_id < 0) ? "input_neurons" : "neurons";
          printf("Presynaptic neurons start index: %d (%s)\n", prestart, presynaptic_group_type_string);
          printf("Presynaptic neurons end index: %d (%s)\n", preend, presynaptic_group_type_string);
          printf("Postsynaptic neurons start index: %d (neurons)\n", poststart);
          printf("Postsynaptic neurons end index: %d (neurons)\n", postend);
	}


	int original_number_of_synapses = total_number_of_synapses;

	// Carry out the creation of the connectivity matrix
	switch (synapse_params->connectivity_type){
            
        case CONNECTIVITY_TYPE_ALL_TO_ALL:
          {
            
            int increment = (preend-prestart)*(postend-poststart);
            this->increment_number_of_synapses(increment);

            // If the connectivity is all_to_all
            for (int i = prestart; i < preend; i++){
              for (int j = poststart; j < postend; j++){
                // Index
                int idx = original_number_of_synapses + (i-prestart)*(postend-poststart) + (j-poststart);
                // Setup Synapses
                presynaptic_neuron_indices[idx] = CORRECTED_PRESYNAPTIC_ID(i, presynaptic_group_is_input);
                postsynaptic_neuron_indices[idx] = j;
              }
            }
            break;
          }
        case CONNECTIVITY_TYPE_ONE_TO_ONE:
          {
            int increment = (preend-prestart);
            this->increment_number_of_synapses(increment);
            
            // If the connectivity is one_to_one
            if ((preend-prestart) != (postend-poststart)) print_message_and_exit("Unequal populations for one_to_one.");
            // Create the connectivity
            for (int i = 0; i < (preend-prestart); i++){
              presynaptic_neuron_indices[original_number_of_synapses + i] = CORRECTED_PRESYNAPTIC_ID(prestart + i, presynaptic_group_is_input);
              postsynaptic_neuron_indices[original_number_of_synapses + i] = poststart + i;
            }

            break;
          }
        case CONNECTIVITY_TYPE_RANDOM: //JI DO
          {
            // If the connectivity is random
            // Begin a count
            for (int i = prestart; i < preend; i++){
              for (int j = poststart; j < postend; j++){
                // Probability of connection
                float prob = ((float)rand() / (RAND_MAX));
                // If it is within the probability range, connect!
                if (prob < synapse_params->random_connectivity_probability){
						
                  this->increment_number_of_synapses(1);

                  // Setup Synapses
                  presynaptic_neuron_indices[total_number_of_synapses - 1] = CORRECTED_PRESYNAPTIC_ID(i, presynaptic_group_is_input);
                  postsynaptic_neuron_indices[total_number_of_synapses - 1] = j;
                }
              }
            }
            break;
          }
		
        case CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE:
          {

            float standard_deviation_sigma = synapse_params->gaussian_synapses_standard_deviation;
            int number_of_new_synapses_per_postsynaptic_neuron = synapse_params->gaussian_synapses_per_postsynaptic_neuron;
			
            int number_of_postsynaptic_neurons_in_group = postend - poststart;
            int total_number_of_new_synapses = number_of_new_synapses_per_postsynaptic_neuron * number_of_postsynaptic_neurons_in_group;
            this->increment_number_of_synapses(total_number_of_new_synapses);

            printf("TODO: FIX set_neuron_indices_by_sampling_from_normal_distribution\n");
            backend()->set_neuron_indices_by_sampling_from_normal_distribution
              ();

            break;
          }
        case CONNECTIVITY_TYPE_SINGLE:
          {
            // If we desire a single connection
            this->increment_number_of_synapses(1);

            // // Setup Synapses
            presynaptic_neuron_indices[original_number_of_synapses] = CORRECTED_PRESYNAPTIC_ID(prestart + int(synapse_params->pairwise_connect_presynaptic), presynaptic_group_is_input);
            postsynaptic_neuron_indices[original_number_of_synapses] = poststart + int(synapse_params->pairwise_connect_postsynaptic);

            break;
          }
        default:
          {
            print_message_and_exit("Unknown Connection Type.");
            break;
          }
	}

	temp_number_of_synapses_in_last_group = total_number_of_synapses - original_number_of_synapses;

	if (print_synapse_group_details == true) printf("%d new synapses added.\n\n", temp_number_of_synapses_in_last_group);

	for (int i = original_number_of_synapses; i < total_number_of_synapses; i++){
		
          float weight_range_bottom = synapse_params->weight_range_bottom;
          float weight_range_top = synapse_params->weight_range_top;

          if (weight_range_bottom == weight_range_top) {
            synaptic_efficacies_or_weights[i] = weight_range_bottom;
          } else {
            float weight = weight_range_bottom + (weight_range_top - weight_range_bottom)*((float)rand() / (RAND_MAX));
			
            synaptic_efficacies_or_weights[i] = weight;
          }

          original_synapse_indices[i] = i;

          // Used for event count
          // printf("postsynaptic_neuron_indices[i]: %d\n", postsynaptic_neuron_indices[i]);
          synapse_postsynaptic_neuron_count_index[postsynaptic_neuron_indices[i]] = neurons->per_neuron_afferent_synapse_count[postsynaptic_neuron_indices[i]];
          neurons->per_neuron_afferent_synapse_count[postsynaptic_neuron_indices[i]] ++;

	}

}

void Synapses::increment_number_of_synapses(int increment) {

  total_number_of_synapses += increment;

  if (total_number_of_synapses - increment == 0) {
          presynaptic_neuron_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
          postsynaptic_neuron_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
          synaptic_efficacies_or_weights = (float*)malloc(total_number_of_synapses * sizeof(float));
          original_synapse_indices = (int*)malloc(total_number_of_synapses * sizeof(int));
          synapse_postsynaptic_neuron_count_index = (int*)malloc(total_number_of_synapses * sizeof(int));
  } else {
    int* temp_presynaptic_neuron_indices = (int*)realloc(presynaptic_neuron_indices, total_number_of_synapses * sizeof(int));
    int* temp_postsynaptic_neuron_indices = (int*)realloc(postsynaptic_neuron_indices, total_number_of_synapses * sizeof(int));
    float* temp_synaptic_efficacies_or_weights = (float*)realloc(synaptic_efficacies_or_weights, total_number_of_synapses * sizeof(float));
    int* temp_original_synapse_indices = (int*)realloc(original_synapse_indices, total_number_of_synapses * sizeof(int));
    int* temp_synapse_postsynaptic_neuron_count_index = (int*)realloc(synapse_postsynaptic_neuron_count_index, total_number_of_synapses * sizeof(int));

    if (temp_presynaptic_neuron_indices != nullptr) presynaptic_neuron_indices = temp_presynaptic_neuron_indices;
    if (temp_postsynaptic_neuron_indices != nullptr) postsynaptic_neuron_indices = temp_postsynaptic_neuron_indices;
    if (temp_synaptic_efficacies_or_weights != nullptr) synaptic_efficacies_or_weights = temp_synaptic_efficacies_or_weights;
    if (temp_original_synapse_indices != nullptr) original_synapse_indices = temp_original_synapse_indices;
    if (temp_synapse_postsynaptic_neuron_count_index != nullptr) synapse_postsynaptic_neuron_count_index = temp_synapse_postsynaptic_neuron_count_index;
  }

}

// Provides order of magnitude speedup for LIF (All to all atleast). 
// Because all synapses contribute to current_injection on every iteration, having all threads in a block accessing only 1 or 2 positions in memory causing massive slowdown.
// Randomising order of synapses means that each block is accessing a larger number of points in memory.
void Synapses::shuffle_synapses() {
  printf("Shuffling synapses...\n");

  std::random_shuffle(&original_synapse_indices[0], &original_synapse_indices[total_number_of_synapses]);

  int* new_presynaptic_neuron_indices = (int *)malloc(total_number_of_synapses*sizeof(int));
  int* new_postsynaptic_neuron_indices = (int *)malloc(total_number_of_synapses*sizeof(int));
  float* new_synaptic_efficacies_or_weights = (float *)malloc(total_number_of_synapses*sizeof(float));
	
  for(int i = 0; i < total_number_of_synapses; i++) {

    new_presynaptic_neuron_indices[i] = presynaptic_neuron_indices[original_synapse_indices[i]];
    new_postsynaptic_neuron_indices[i] = postsynaptic_neuron_indices[original_synapse_indices[i]];
    new_synaptic_efficacies_or_weights[i] = synaptic_efficacies_or_weights[original_synapse_indices[i]];

  }

  presynaptic_neuron_indices = new_presynaptic_neuron_indices;
  postsynaptic_neuron_indices = new_postsynaptic_neuron_indices;
  synaptic_efficacies_or_weights = new_synaptic_efficacies_or_weights;

}

MAKE_STUB_PREPARE_BACKEND(Synapses);
