/* \brief The most abstract Synapses class from which methods and attributes are inherited by SpikingSynapses etc.
*/

#ifndef SYNAPSES_H
#define SYNAPSES_H

/**
 * @file   Synapses.h
 * @brief  The most abstract Synapses class from which methods and attributes are inherited by SpikingSynapses etc.
 *
 */

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"
#include "Spike/Plasticity/Plasticity.hpp"
#include "Spike/Neurons/Neurons.hpp"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>
#include <vector>

#include "Spike/Helpers/RandomStateManager.hpp"

class Synapses; // forward definition

namespace Backend {
  class Synapses : public virtual SpikeBackendBase  {
  public:
    SPIKE_ADD_BACKEND_FACTORY(Synapses);
    ~Synapses() override = default;
    virtual void set_neuron_indices_by_sampling_from_normal_distribution
    (int original_number_of_synapses,
     int total_number_of_new_synapses,
     int postsynaptic_group_id,
     int poststart, int prestart,
     int* postsynaptic_group_shape,
     int* presynaptic_group_shape,
     int number_of_new_synapses_per_postsynaptic_neuron,
     int number_of_postsynaptic_neurons_in_group,
     int max_number_of_connections_per_pair,
     float standard_deviation_sigma,
     bool presynaptic_group_is_input) = 0;
  };
}

static_assert(std::has_virtual_destructor<Backend::Synapses>::value, "contract violated");

/*!
  This enum contains the possible connectivity types which can be requested
  when adding synapstic connections between neuron populations.
*/
enum CONNECTIVITY_TYPE
{
  CONNECTIVITY_TYPE_ALL_TO_ALL,
  CONNECTIVITY_TYPE_ONE_TO_ONE,
  CONNECTIVITY_TYPE_RANDOM,
  CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE,
  CONNECTIVITY_TYPE_SINGLE
};

/*!
  This struct accompanies the synapses class and is used to provide the a number
  of parameters to control synapse creation.
  This is also used as the foundation for the more sophisticated neuron parameter structs
  for SpikingSynapses etc.
*/
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
  std::vector<Plasticity*> plasticity_vec;
};

/*!
  This is the parent class for SpikingSpiking.
  It provides a set of default methods which are primarily used to add groups of synapses to the connectivity.
*/
class Synapses : public virtual SpikeBase {

public:
  // Constructor/Destructor
  Synapses();
  Synapses(int seedval);
  ~Synapses() override;

  SPIKE_ADD_BACKEND_GETSET(Synapses, SpikeBase);
  void init_backend(Context* ctx = _global_ctx) override;
  void prepare_backend_early() override;
  
  // Variables
  int total_number_of_synapses = 0;                 /**< Tracks the total number of synapses in the connectivity */
  int temp_number_of_synapses_in_last_group = 0;    /**< Tracks the number of synapses in the last added group. */
  int largest_synapse_group_size = 0;               /**< Tracks the size of the largest synaptic group. */
  bool print_synapse_group_details = false;         /**< A flag used to indicate whether group details should be printed */
  std::vector<Plasticity*> plasticity_rule_vec;     /**< A vector of pointers to the plasticity rules to be used in the simulation */
  std::vector<int> plasticity_synapse_number_per_rule;  /**< A vector of the number of synapse ids to be associated with each plasticity rule */
  

	
  // Host Pointers
  int* presynaptic_neuron_indices = nullptr;                /**< Indices of presynaptic neuron IDs */
  int* postsynaptic_neuron_indices = nullptr;               /**< Indices of postsynaptic neuron IDs */
  int* original_synapse_indices = nullptr;                  /**< Indices by which to order the presynaptic and postsynaptic neuron IDs if a shuffle is carried out */
  int* synapse_postsynaptic_neuron_count_index = nullptr;   /**< An array of the number of incoming synapses to each postsynaptic neuron */
  int maximum_number_of_afferent_synapses = 0;
  float* synaptic_efficacies_or_weights = nullptr;          /**< An array of synaptic efficacies/weights accompanying the pre/postsynaptic_neuron_indices */
  std::vector<int*> plasticity_synapse_indices_per_rule;    /**< A vector (host-side) which contains the list of synapse ids that each corresponding plasticity rule must be applied to */
  

  // Functions

  /**
   *  Determines the synaptic connections to add to the simulation connectivity.
   This is a virtual function to allow polymorphism in the methods of various sub-class implementations.
    Allocates memory as necessary for group size and indices storage.
  \param presynaptic_group_id An int ID of the presynaptic neuron population (ID is < 0 if presynaptic population is an InputSpikingNeuron population)
  \param postsynaptic_group_id An int ID of the postsynaptic neuron population.
  \param neurons A pointer to an instance of the Neurons class (or sub-class) which is included in this simulation.
  \param input_neurons A pointer to an instance of the Neurons class (or sub-class) which is included in this simulation (for population indices < 0, i.e. InputSpikingNeurons).
  \param timestep A float indicating the timestep at which the simulator is running.
  \param synapse_params A synapse_parameters_struct pointer describing the connectivity arrangement to be added between the pre and postsynaptic neuron populations.
   */
  virtual void AddGroup(int presynaptic_group_id, 
                        int postsynaptic_group_id, 
                        Neurons * neurons,
                        Neurons * input_neurons,
                        float timestep,
                        synapse_parameters_struct * synapse_params);

  /**
     *  A function called in to reallocate memory for a given number more synapses.
     /param increment The number of synapses for which allocated memory must be expanded.
  */
  void increment_number_of_synapses(int increment);

  /**
     *  A function to shuffle the order of synapse array storage for increased GPU memory write speeds. Currently Unused.
  */
  virtual void shuffle_synapses();

  void reset_state() override;

  RandomStateManager * random_state_manager;

private:
  std::shared_ptr<::Backend::Synapses> _backend;
};

// GAUSS random number generator
double randn (double mu, double sigma);
#endif
