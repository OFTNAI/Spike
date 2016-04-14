#include "Inputs.h"
#include <stdlib.h>
#include <stdio.h>
#include "CUDAErrorCheckHelpers.h"
#include <algorithm> // For random shuffle
using namespace std;


// Inputs Constructor
Inputs::Inputs() {

	// Set totals to zero
	total_number_of_inputs = 0;
	total_number_of_groups = 0;

	// Initialise pointers
	group_shapes = NULL;
	input_variables = NULL;
	last_input_indices_for_each_group = NULL;

}


// Inputs Destructor
Inputs::~Inputs() {

	// Free up memory
	free(input_variables);
	free(input_variables);
	free(last_input_indices_for_each_group);

	CudaSafeCall(cudaFree(d_input_variables));
	CudaSafeCall(cudaFree(d_lastspiketime));

}


int Inputs::AddGroup(input_struct params, int group_shape[2]){
	
	int number_of_inputs_in_group = group_shape[0]*group_shape[1];
 
	if (number_of_inputs_in_group < 0) {
		printf("\nError: Group must have at least 1 inputs.\n\n");
		exit(-1);
	}

	// Update totals
	total_number_of_inputs += number_of_inputs_in_group;
	++total_number_of_groups;
	printf("total_number_of_groups: %d\n", total_number_of_groups); // Temp helper

	// Calculate new group id
	int new_group_index = total_number_of_groups - 1;

	// Add last input index for new group
	last_input_indices_for_each_group = (int*)realloc(last_input_indices_for_each_group,(total_number_of_groups*sizeof(int)));
	last_input_indices_for_each_group[new_group_index] = total_number_of_inputs;

	// Add new group shape
	group_shapes = (int**)realloc(group_shapes,(total_number_of_groups*sizeof(int*)));
	group_shapes[new_group_index] = (int*)malloc(2*sizeof(int));
	group_shapes[new_group_index] = group_shape;

	// Add new group parameters
	input_variables = (input_struct*)realloc(input_variables, (total_number_of_inputs*sizeof(input_struct)));
	for (int i = (total_number_of_inputs - number_of_inputs_in_group); i < total_number_of_inputs; i++){
		input_variables[i] = params;
	}
	
	return -total_number_of_groups;
}

void Inputs::generate_states() {
	cudaMalloc((void**) &d_states, total_number_of_inputs*sizeof(curandState_t));
	// Initialise the random states
	init<<<threads_per_block, number_of_input_blocks_per_grid>>>(42, d_states, total_number_of_inputs);
	CudaCheckError();
}


void Inputs::initialise_device_pointers() {
	CudaSafeCall(cudaMalloc((void **)&d_input_variables, sizeof(struct input_struct)*total_number_of_inputs));
	// CudaSafeCall(cudaMalloc((void **)&d_lastspiketime, sizeof(float)*total_number_of_inputs));

	reset_input_variables();
}

void Inputs::reset_input_variables() {
	CudaSafeCall(cudaMemcpy(d_input_variables, input_variables, sizeof(struct input_struct)*total_number_of_inputs, cudaMemcpyHostToDevice));
// 	// CudaSafeCall(cudaMemset(d_lastspiketime, -1000.0f, total_number_of_neurons*sizeof(float)));
}

void Inputs::set_threads_per_block_and_blocks_per_grid(int threads) {
	
	threads_per_block.x = threads;

	int number_of_input_blocks = (total_number_of_inputs + threads) / threads;
	number_of_input_blocks_per_grid.x = number_of_input_blocks;
}


__global__ void poisupdate2(curandState_t* states,
							struct input_struct* d_input_group_variables,
							float timestep,
							size_t total_number_of_inputs);


// // Wrapper member function definitions
// // See NOTE above
void Inputs::poisupdate_wrapper2(float timestep) {

	poisupdate2<<<number_of_input_blocks_per_grid, threads_per_block>>>(d_states,
														d_input_variables,
														timestep,
														total_number_of_inputs);
	CudaCheckError();
}


// Random Number Generator intialiser
/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states, size_t total_number_of_neurons) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_neurons) {
		curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
					idx, /* the sequence number should be different for each core (unless you want all
							cores to get the same sequence of numbers for some reason - use thread id! */
 					0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
					&states[idx]);
	}
}

// // CUDA __global__ function definitions
// // These are called by the Neurons class member functions
// // May have to vary names if 'including' more than one subclass

// Poisson Updating Kernal
__global__ void poisupdate2(curandState_t* states,
							struct input_struct* d_input_group_variables,
							float timestep,
							size_t total_number_of_inputs){

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < total_number_of_inputs){

		float random_float = curand_uniform(&states[idx]);;
		
		// if the randomnumber is LT the rate
		if (random_float < (d_input_group_variables[idx].rate*timestep)){
			d_input_group_variables[idx].state_u = 0.0f;
			d_input_group_variables[idx].state_v = 35.0f;
		} else if (d_input_group_variables[idx].rate != 0.0f) {
			d_input_group_variables[idx].state_u = 0.0f;
			d_input_group_variables[idx].state_v = -70.0f;
		}
	}
	__syncthreads();
}

