/* \brief The most abstract Neuron class from which methods and attributes are inherited by SpikingNeurons etc.
*/
#ifndef Neurons_H
#define Neurons_H

/**
 * @file   Neurons.h
 * @brief  The most abstract Neuron class from which methods and attributes are inherited by SpikingNeurons etc.
 *
 */

#include "Spike/Base.hpp"

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

#include <cstdio>

class Neurons; // forward definition

namespace Backend {
  class Neurons : public virtual SpikeBackendBase {
  public:
    SPIKE_ADD_BACKEND_FACTORY(Neurons);
    ~Neurons() override = default;
  };
}

static_assert(std::has_virtual_destructor<Backend::Neurons>::value,
              "contract violated");

#define PRESYNAPTIC_IS_INPUT( id ) (id < 0 ? true : false)
#define CORRECTED_PRESYNAPTIC_ID(id, is_input) (is_input ? (-1 * (id)) - 1 : id) 

/*!
	This struct accompanies the neuron class and is used to provide the number of neurons 
	(in a 2D grid) that are to be added to the total population.
	This is also used as the foundation for the more sophisticated neuron parameter structs 
	for SpikingNeurons etc.
*/
struct neuron_parameters_struct {
	neuron_parameters_struct() {}

	int group_shape[2];		/**< An int array with 2 values describing the 2D shape of a neuron group */
};

/*!
  This is the parent class for SpikingNeurons.
  It provides a set of default methods which are primarily used to add groups of neurons to the total population.
*/
class Neurons : public virtual SpikeBase {
public:  
  Neurons();
  ~Neurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(Neurons, SpikeBase);
  
  // Variables
  int total_number_of_neurons;				/**< Tracks the total neuron population size. */
  int total_number_of_groups;					/**< Tracks the number of groups (the total neuron population is split into groups e.g. layers or excitatory/inh). */
  int number_of_neurons_in_new_group;			/**< Stores number of neurons in most recently added group */

  // Host Pointers
  int *start_neuron_indices_for_each_group;	/**< Indices of the beginnings of each group in the total population. */
  int *last_neuron_indices_for_each_group;	/**< Indices of the final neuron in each group. */
  int * per_neuron_afferent_synapse_count;	/**< A (host-side) count of the number of afferent synapses for each neuron */
  int **group_shapes;							/**< The 2D shape of each group. */

  int *per_neuron_efferent_synapse_count;	/**< An array containing the number of output synapses from each neuron */
  int **per_neuron_efferent_synapse_indices;	/**< A 2D array detailing the synapse indices of efferent synapses from each neuron */
  
  /**  
   *  Determines the total number of neurons by which the simulation should increase.
   This is a virtual function to allow polymorphism in the methods of various SpikingNeuron implementations.
   Allocates memory as necessary for group size and indices storage.
   \param group_params A neuron_parameters_struct instance describing a 2D neuron population size.
   \return The unique ID for the population which was requested for creation.
  */
  virtual int AddGroup(neuron_parameters_struct * group_params);

  /**  
   *  Resets any undesired data which is dynamically reassigned during a simulation. 
   */
  void reset_state() override;


  /**
   * Adds efferent synapse IDs to the per neuron efferent synapse groiups
   \param neuron_id the id of the presynaptic neuron from which the synapse emerges
   \param synapse_id the id of the efferent synapse
  */
  void AddEfferentSynapse(int neuron_id, int synapse_id);

private:
  std::shared_ptr<::Backend::Neurons> _backend;
};

#endif
