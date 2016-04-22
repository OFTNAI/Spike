#ifndef PoissonSpikingNeurons_H
#define PoissonSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SpikingNeurons.h"

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

	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void initialise_device_pointers();
	virtual void reset_neurons();

	void generate_random_states();
	void update_poisson_states(float timestep);

};

#endif