// Synapse Class Header
// Synapse.h
//
//	Author: Nasir Ahmad
//	Date: 7/12/2015

#ifndef SYNAPSES_H
#define SYNAPSES_H

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

namespace Backend {
  /* Each type T deriving from ::Synapses must have an associated
   * Backend::T class specifying the interface implementing the actual
   * computations. Different backends then specialize this
   * Backend::T. For example, Backend::CUDA::T implements the
   * computations required by T, as specified by Backend::T, for the
   * CUDA backend.
   * 
   * Whilst the Backend::T classes mirror the ::T hierarchy, the
   * Backend::CUDA::T type only directly inherits from
   * Backend::T. Methods shared by different ::Synapses types are
   * implemented in a parallel hierarchy, deriving from the virtual
   * class Backend::SynapsesCommon (which specifies the interface for
   * the common computations).  Then, Backend::CUDA::SynapsesCommon
   * (for example) implements these common computations, and other
   * Backend::CUDA::T classes derive from it. The inheritance in this
   * case is virtual, to avoid the `diamond problem' for pure virtual
   * methods.
   * 
   * A good example of this structure is given by ConductanceSpikingSynapses.
   */
  class SynapsesCommon {
  public:
    virtual void set_neuron_indices_by_sampling_from_normal_distribution() = 0;
  };

  class Synapses : public virtual SynapsesCommon,
                   public Generic  {
  public:
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
  ADD_BACKEND_GETTER(Synapses);
  
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
  virtual void prepare_backend(Context* ctx = _global_ctx);
  virtual void reset_state() = 0;

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
