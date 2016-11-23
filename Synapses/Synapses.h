/* \brief The most abstract Synapses class from which methods and attributes are inherited by SpikingSynapses etc.
*/
#ifndef SYNAPSES_H
#define SYNAPSES_H

/**
 * @file   Synapses.h
 * @brief  The most abstract Synapses class from which methods and attributes are inherited by SpikingSynapses etc.
 *
 */

#include "../Neurons/Neurons.h"

// stdlib allows random numbers
#include <stdlib.h>
// Input Output
#include <stdio.h>
// allows maths
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

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
	synapse_parameters_struct(): max_number_of_connections_per_pair(1), gaussian_synapses_per_postsynaptic_neuron(10), gaussian_synapses_standard_deviation(10.0), weight_range_bottom(0.0), weight_range_top(1.0), connectivity_type(CONNECTIVITY_TYPE_ALL_TO_ALL)  {}

	int max_number_of_connections_per_pair;  /**< Used in specific branches to indicate max number of connections between a specific pair of pre and postsynaptic neurons. UNUSED IN THIS BRANCH.*/
	int pairwise_connect_presynaptic;  /**< If using CONNECTIVITY_TYPE_ONE_TO_ONE, should be set to the presynaptic neuron index. Else, unused.*/
	int pairwise_connect_postsynaptic; /**< If using CONNECTIVITY_TYPE_ONE_TO_ONE, should be set to the postsynaptic neuron index. Else, unused.*/
	int gaussian_synapses_per_postsynaptic_neuron; /**< If using CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, should be set to the max number of desired connections to each postsynaptic neuron. Else, unused.*/
	float gaussian_synapses_standard_deviation;  /**< If using CONNECTIVITY_TYPE_GAUSSIAN_SAMPLE, should be set to the desired standard deviation of the gaussian distribution to be used to select presynaptic neurons. Else, unused.*/
	float weight_range_bottom; /**< Used for all CONNECTIVITY types. Indicates the lower range (inclusive) of the weight range to be used */
	float weight_range_top;  /**< Used for all CONNECTIVITY types. Indicates the upper range (inclusive) of the weight range to be used. If weight_range_bottom == weight_range_top, all weight values are equal. Else, they are a random uniform distribution in this range*/
	float random_connectivity_probability; /**< If using CONNECTIVITY_TYPE_RANDOM, indicates the probability with which a connection should be made between the pre and post synaptic neurons. Else, unused.*/
	int connectivity_type; /**< A CONNECTIVITY_TYPE is set here */

};

/*!
	This is the parent class for SpikingSpiking.
	It provides a set of default methods which are primarily used to add groups of synapses to the connectivity.
*/
class Synapses {

public:

	// Constructor/Destructor
	Synapses();
	~Synapses();


	// Variables
	int total_number_of_synapses;                  /**< Tracks the total number of synapses in the connectivity */
	int temp_number_of_synapses_in_last_group;     /**< Tracks the number of synapses in the last added group. */
	int largest_synapse_group_size;                /**< Tracks the size of the largest synaptic group. */
	bool print_synapse_group_details;              /**< A flag used to indicate whether group details should be printed */

	// Host Pointers
	int* presynaptic_neuron_indices;               /**< Indices of presynaptic neuron IDs */
	int* postsynaptic_neuron_indices;              /**< Indices of postsynaptic neuron IDs */
	int* original_synapse_indices;                 /**< Indices by which to order the presynaptic and postsynaptic neuron IDs if a shuffle is carried out */
	int* synapse_postsynaptic_neuron_count_index;  /**< An array of the number of incoming synapses to each postsynaptic neuron */
	float* synaptic_efficacies_or_weights;         /**< An array of synaptic efficacies/weights accompanying the pre/postsynaptic_neuron_indices */

	// Device pointers
	int* d_presynaptic_neuron_indices;             /**< A (device-side) pointer to store indices of the presynaptic neuron IDs */
	int* d_postsynaptic_neuron_indices;            /**< A (device-side) pointer to store indices of the postsynaptic neuron IDs */
	int* d_temp_presynaptic_neuron_indices;
	int* d_temp_postsynaptic_neuron_indices;
	int * d_synapse_postsynaptic_neuron_count_index; /**< A (device-side) pointer to store the number of incoming synapses to each neuron */
	float* d_synaptic_efficacies_or_weights;         /**< A (device-side) pointer to store synaptic efficaces/weights accompanying pre/postsynaptic neuron indices */
	float* d_temp_synaptic_efficacies_or_weights;

	// CUDA Specific
	dim3 number_of_synapse_blocks_per_grid;        /**< A CUDA type, storing the number of blocks per grid for synapse GPU operations */
	dim3 threads_per_block;                        /**< A CUDA type, storing the number of threads per block for synapse GPU operations */

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
     *  Exclusively for the allocation of device memory. This class allocates device pointers for presynaptic and postsynaptic neuron ids, synaptic efficacies/weights, and the number of synaptic connections to the postsynaptic neuron.
     */
	virtual void allocate_device_pointers();

  /**
     *  Allows copying of static data related to neuron dynamics to the device. Copies synapse pre/post ids, efficies/weights, and number of postsynaptic contacts on postsynaptic neuron.
	*/
	virtual void copy_constants_and_initial_efficacies_to_device();

  /**
     *  A local, function called in to determine the CUDA Device thread (Synapses::threads_per_block) and block dimensions (Synapses::number_of_synapse_blocks_per_grid).
	*/
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);

  /**
     *  A function called in to reallocate memory for a given number more synapses.
     /param increment The number of synapses for which allocated memory must be expanded.
  */
	virtual void increment_number_of_synapses(int increment);

  /**
     *  A function to shuffle the order of synapse array storage for increased GPU memory write speeds. Currently Unused.
  */
	virtual void shuffle_synapses();




};

__global__ void compute_yes_no_connection_matrix_for_groups(bool * d_yes_no_connection_vector,
														int pre_width,
														int post_width,
														int post_height,
														float sigma,
														int total_pre_neurons,
														int total_post_neurons);

__global__ void set_up_neuron_indices_and_weights_for_yes_no_connection_matrix(bool * d_yes_no_connection_vector,
																			int pre_width,
																			int post_width,
																			int post_height,
																			int total_pre_neurons,
																			int total_post_neurons,
																			int * d_presynaptic_neuron_indices,
																			int * d_postsynaptic_neuron_indices);

__global__ void set_neuron_indices_by_sampling_from_normal_distribution(int total_number_of_new_synapses,
																		int postsynaptic_group_id,
																		int poststart,
																		int prestart,
																		int post_width,
																		int post_height,
																		int pre_width,
																		int pre_height,
																		int number_of_new_synapses_per_postsynaptic_neuron,
																		int number_of_postsynaptic_neurons_in_group,
																		int * d_presynaptic_neuron_indices,
																		int * d_postsynaptic_neuron_indices,
																		float * d_synaptic_efficacies_or_weights,
																		float standard_deviation_sigma,
																		bool presynaptic_group_is_input,
																		curandState_t* d_states);

// GAUSS random number generator
double randn (double mu, double sigma);
#endif
