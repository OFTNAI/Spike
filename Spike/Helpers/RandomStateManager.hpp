#ifndef RANDOMSTATEMANAGER_H
#define RANDOMSTATEMANAGER_H

//CUDA #include <cuda.h>
//CUDA #include <curand.h>
//CUDA #include <curand_kernel.h>
#include "Spike/CUDA_Hacks.hpp"




class RandomStateManager {

public:

	// Constructor/Destructor
	RandomStateManager();
	~RandomStateManager();

	//CUDA curandState_t* d_states;
        void* d_states;
	int total_number_of_states;

	dim3 threads_per_block;
	dim3 block_dimensions;

	void setup_random_states(int threads_per_blocks_x = 128, int number_of_blocks_x = 64, int seed = 1);

private:
    static RandomStateManager *inst;

	
};

#endif
