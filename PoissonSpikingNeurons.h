#ifndef PoissonSpikingNeurons_H
#define PoissonSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SpikingNeurons.h"
#include "Connections.h"

struct poisson_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	poisson_spiking_neuron_parameters_struct(): rate(0.0f) { spiking_neuron_parameters_struct(); }

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
	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void initialise_device_pointers();
	void reset_input_variables();

	void generate_random_states_wrapper();
	void update_poisson_state_wrapper(float timestep);

};

__global__ void init(unsigned int seed, curandState_t* states, size_t numNeurons);


#endif