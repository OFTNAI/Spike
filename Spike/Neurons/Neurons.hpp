/* \brief The most abstract Neuron class from which methods and attributes are inherited by SpikingNeurons etc.
*/
#ifndef Neurons_H
#define Neurons_H

/**
 * @file   Neurons.h
 * @brief  The most abstract Neuron class from which methods and attributes are inherited by SpikingNeurons etc.
 *
 */

//CUDA #include <cuda.h>
//CUDA #include <vector_types.h>
#include "Spike/CUDA_Hacks.hpp"
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
public:/**
     *  Initializes pointers and variables.
     */
	
	Neurons();
	~Neurons();

	// Variables
	int total_number_of_neurons;				/**< Tracks the total neuron population size. */
	int total_number_of_groups;					/**< Tracks the number of groups (the total neuron population is split into groups e.g. layers or excitatory/inh). */
	int number_of_neurons_in_new_group;			/**< Stores number of neurons in most recently added group */

	// Host Pointers
	int *start_neuron_indices_for_each_group;	/**< Indices of the beginnings of each group in the total population. */
	int *last_neuron_indices_for_each_group;	/**< Indices of the final neuron in each group. */
	int * per_neuron_afferent_synapse_count;	/**< A (host-side) count of the number of afferent synapses for each neuron */
	int **group_shapes;							/**< The 2D shape of each group. */

	// Device Pointers
	int * d_per_neuron_afferent_synapse_count;	/**< A (device-side) count of the number of afferent synapses for each neuron */
	float* d_current_injections;				/**< Device array for the storage of current to be injected into each neuron on each timestep. */

	// CUDA Specific
	dim3 number_of_neuron_blocks_per_grid;		/**< CUDA Device number of blocks */
	dim3 threads_per_block;						/**< CUDA Device number of threads */

	// Functions
	/**  
     *  Determines the total number of neurons by which the simulation should increase.
     This is a virtual function to allow polymorphism in the methods of various SpikingNeuron implementations.
     	Allocates memory as necessary for group size and indices storage.
		\param group_params A neuron_parameters_struct instance describing a 2D neuron population size.
		\return The unique ID for the population which was requested for creation.
     */
	virtual int AddGroup(neuron_parameters_struct * group_params);

	/**  
     *  Exclusively for the allocation of device memory. This class requires allocation of d_current_injections only.
		\param maximum_axonal_delay_in_timesteps The length (in timesteps) of the largest axonal delay in the simulation. Unused in this class.
		\param high_fidelity_spike_storage A flag determining whether a bit mask based method is used to store spike times of neurons (ensure no spike transmission failure).
     */
	virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
	
	/**  
     *  Unused in this class. Allows copying of static data related to neuron dynamics to the device.
	*/
	virtual void copy_constants_to_device();

	/**  
     *  Resets any undesired device based data which is dynamically reassigned during a simulation. In this case, the current to be injected at the next time step.
	*/
	virtual void reset_neuron_activities();
	
	/**  
     *  A local, non-polymorphic function called by Neurons::reset_neuron_activities to reset Neurons::d_current_injections.
	*/
	void reset_current_injections();

	/**  
     *  A local, non-polymorphic function called in to determine the CUDA Device thread (Neurons::threads_per_block) and block dimensions (Neurons::number_of_neuron_blocks_per_grid).
	*/
	void set_threads_per_block_and_blocks_per_grid(int threads);
};

#endif
