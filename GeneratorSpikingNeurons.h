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


	
	// Functions
	virtual int AddGroupNew(neuron_struct * params, int shape[2]);
	virtual void initialise_device_pointersNew();
	void reset_input_variables();


	void generupdate2_wrapper(int* genids,
							float* gentimes,
							float currtime,
							float timestep,
							size_t numEntries);

};

#endif