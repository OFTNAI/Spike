#ifndef PoissonSpikingNeurons_H
#define PoissonSpikingNeurons_H

//	CUDA library
#include <cuda.h>

#include <curand.h>
#include <curand_kernel.h>

//temp for test_array test
#include "Connections.h"

#include "SpikingNeurons.h"

struct poisson_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	poisson_spiking_neuron_parameters_struct(): rate(0.0f) { spiking_neuron_parameters_struct(); }

	float rate;
};

struct input_struct {

	input_struct(): paramc(0.0f), paramd(0.0f), state_v(-70.0f), state_u(0.0f), rate(0.0f) { }   // default Constructor
	float paramc;
	float paramd;
	// State variables
	float state_v;
	float state_u;
	// Rate for poisson
	float rate;
};

class PoissonSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	PoissonSpikingNeurons();
	~PoissonSpikingNeurons();

	float * rates;
	float * d_rates;

	curandState_t* d_states;

	
	// Functions
	virtual int AddGroupNew(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void initialise_device_pointersNew();
	void reset_input_variables();

	void generate_random_states_wrapper();
	void update_poisson_state_wrapper(float timestep);

};

__global__ void init(unsigned int seed, curandState_t* states, size_t numNeurons);


#endif