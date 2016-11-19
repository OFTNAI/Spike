/* \brief The most abstract Neuron class from which methods and attributes are inherited by SpikingNeurons etc.
*/
#ifndef Neurons_H
#define Neurons_H

/**
 * @file   Neurons.h
 * @brief  The most abstract Neuron class from which methods and attributes are inherited by SpikingNeurons etc.
 *
 */

#include "Spike/Backend/Macros.hpp"
#include "Spike/Backend/Context.hpp"
#include "Spike/Backend/Backend.hpp"
#include "Spike/Backend/Device.hpp"

namespace Backend {
  class Neurons : public Generic {
  public:
    virtual void reset_state();
    virtual void prepare();
  };
}

#include "Spike/Backend/Dummy/Neurons/Neurons.hpp"

#include <stdio.h>


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
class Neurons{
public:  
  Neurons();
  ~Neurons();

  Backend::Neurons* backend;
  
  // Variables
  int total_number_of_neurons;				/**< Tracks the total neuron population size. */
  int total_number_of_groups;					/**< Tracks the number of groups (the total neuron population is split into groups e.g. layers or excitatory/inh). */
  int number_of_neurons_in_new_group;			/**< Stores number of neurons in most recently added group */

  // Host Pointers
  int *start_neuron_indices_for_each_group;	/**< Indices of the beginnings of each group in the total population. */
  int *last_neuron_indices_for_each_group;	/**< Indices of the final neuron in each group. */
  int * per_neuron_afferent_synapse_count;	/**< A (host-side) count of the number of afferent synapses for each neuron */
  int **group_shapes;							/**< The 2D shape of each group. */

  // Functions
  virtual void prepare_backend(Context* ctx);
  
  /**  
   *  Determines the total number of neurons by which the simulation should increase.
   This is a virtual function to allow polymorphism in the methods of various SpikingNeuron implementations.
   Allocates memory as necessary for group size and indices storage.
   \param group_params A neuron_parameters_struct instance describing a 2D neuron population size.
   \return The unique ID for the population which was requested for creation.
  */
  virtual int AddGroup(neuron_parameters_struct * group_params);

  /**  
   *  Resets any undesired data which is dynamically reassigned during a simulation. In this case, the current to be injected at the next time step.
   */
  virtual void reset_state();
};

#endif
