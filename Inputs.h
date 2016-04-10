#ifndef Inputs_H
#define Inputs_H

//	CUDA library
#include <cuda.h>

#include "Structs.h"

//temp for test_array test
#include "Connections.h"

class Inputs {
public:
	// Constructor/Destructor
	Inputs();
	~Inputs();

	// Totals
	int total_number_of_inputs;
	int total_number_of_groups;

	// Group parameters, shapes and indices
	input_struct *input_variables;
	int **group_shapes;
	int *last_input_indices_for_each_group;


	// Device Pointers
	input_struct* d_input_variables;
	float* d_lastspiketime;

	dim3 number_of_input_blocks_per_grid;
	dim3 threads_per_block;

	
	// Functions
	int AddGroup(input_struct params, int shape[2]);

	void initialise_device_pointers();
	// void reset_input_variables_and_spikes();
	void set_threads_per_block_and_blocks_per_grid(int threads);

	// void poisupdate_wrapper(float* d_randoms, float timestep);

	// void genupdate_wrapper(int* genids,
	// 						float* gentimes,
	// 						float currtime,
	// 						float timestep,
	// 						size_t numEntries,
	// 						int genblocknum, 
	// 						dim3 threadsPerBlock);

	// void spikingneurons_wrapper(float currtime);

	// void stateupdate_wrapper(float* current_injection,
	// 						float timestep);



};



#endif