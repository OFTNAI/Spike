#ifndef GeneratorSpikingNeurons_H
#define GeneratorSpikingNeurons_H

//	CUDA library
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

//temp for test_array test
#include "Connections.h"

#include "SpikingNeurons.h"


class GeneratorSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	GeneratorSpikingNeurons();
	~GeneratorSpikingNeurons();

	int** genids;
	float** gentimes;

	int* d_genids;
	float* d_gentimes;

	int numEnts;
	dim3 genblocksPerGrid;
	
	// Functions
	virtual int AddGroupNew(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void initialise_device_pointers_for_ents(int numEnts, int present);
	virtual void reset_input_variables(int present);

	virtual void set_threads_per_block_and_blocks_per_grid(int threads);

	void generupdate2_wrapper(float currtime,
							float timestep);

};

#endif