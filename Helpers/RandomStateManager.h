#ifndef RANDOMSTATEMANAGER_H
#define RANDOMSTATEMANAGER_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>





class RandomStateManager {

public:

	// Constructor/Destructor
	RandomStateManager();
	~RandomStateManager();

	curandState_t* d_states = NULL;

	dim3 threads_per_block;
	dim3 block_dimensions;

	int total_number_of_states;

	void set_up_random_states(int threads_per_blocks_x, int number_of_blocks_x, int seed);

	static RandomStateManager* instance();

private:
    static RandomStateManager *inst;

	
};

#endif