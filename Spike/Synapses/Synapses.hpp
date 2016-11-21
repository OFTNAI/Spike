// Synapse Class Header
// Synapse.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef SYNAPSES_H
#define SYNAPSES_H

#include "../Neurons/Neurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"

#include "../Helpers/RandomStateManager.hpp"

namespace Backend {
  class Synapses : public Generic {
  public:
    virtual void reset_state() = 0;
    virtual void prepare() = 0;
  };
}

#include "Spike/Backend/Dummy/Synapses/Synapses.hpp"

enum CONNECTIVITY_TYPE
{
  CONNECTIVITY_TYPE_ALL_TO_ALL,
  CONNECTIVITY_TYPE_ONE_TO_ONE,
  CONNECTIVITY_TYPE_RANDOM,
  CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE,
  CONNECTIVITY_TYPE_SINGLE
};

struct synapse_parameters_struct {
  int max_number_of_connections_per_pair = 1;
  int pairwise_connect_presynaptic;
  int pairwise_connect_postsynaptic;
  int gaussian_synapses_per_postsynaptic_neuron = 10;
  float gaussian_synapses_standard_deviation = 10.0;
  float weight_range_bottom = 0.0;
  float weight_range_top = 1.0;
  float random_connectivity_probability;
  int connectivity_type = CONNECTIVITY_TYPE_ALL_TO_ALL;
};


class Synapses {

public:
  // Constructor/Destructor
  Synapses();
  ~Synapses();

  void* _backend;
  
  // Variables
  int total_number_of_synapses = 0;
  int temp_number_of_synapses_in_last_group = 0;
  int largest_synapse_group_size = 0;
  bool print_synapse_group_details = false;
	
  // Host Pointers
  int* presynaptic_neuron_indices = NULL;
  int* postsynaptic_neuron_indices = NULL; 
  int* original_synapse_indices = NULL;
  int* synapse_postsynaptic_neuron_count_index = NULL;
  float* synaptic_efficacies_or_weights = NULL;

  // Functions
  void prepare_backend(Context* ctx) {
    printf("TODO: Synapse prepare_backend\n");
  }

  void reset_state() {
    printf("TODO: Synapse reset_state\n");
  }

  virtual void AddGroup(int presynaptic_group_id, 
                        int postsynaptic_group_id, 
                        Neurons * neurons,
                        Neurons * input_neurons,
                        float timestep,
                        synapse_parameters_struct * synapse_params);

  virtual void increment_number_of_synapses(int increment);
  virtual void shuffle_synapses();

protected:
  RandomStateManager * random_state_manager;
};

// GAUSS random number generator
double randn (double mu, double sigma);
#endif
