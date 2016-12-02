// Synapse Class Header
// Synapse.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef SYNAPSES_H
#define SYNAPSES_H

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include "Spike/Neurons/Neurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include "Spike/Helpers/RandomStateManager.hpp"

class Synapses; // forward definition

namespace Backend {
  class Synapses : public virtual SpikeBackendBase  {
  public:
    virtual void set_neuron_indices_by_sampling_from_normal_distribution() = 0;
  };
}

#include "Spike/Backend/Dummy/Synapses/Synapses.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Synapses/Synapses.hpp"
#endif

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


class Synapses : public virtual SpikeBase {

public:
  // Constructor/Destructor
  Synapses();
  ~Synapses();

  ADD_BACKEND_GETTER(Synapses);
  virtual void prepare_backend(Context* ctx = _global_ctx);
  virtual void prepare_backend_extra();
  virtual void reset_state() = 0;
  
  // Variables
  int total_number_of_synapses = 0;
  int temp_number_of_synapses_in_last_group = 0;
  int largest_synapse_group_size = 0;
  bool print_synapse_group_details = false;
	
  // Host Pointers
  int* presynaptic_neuron_indices = nullptr;
  int* postsynaptic_neuron_indices = nullptr; 
  int* original_synapse_indices = nullptr;
  int* synapse_postsynaptic_neuron_count_index = nullptr;
  float* synaptic_efficacies_or_weights = nullptr;

  // Functions
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
