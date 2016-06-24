#ifndef PoissonInputSpikingNeurons_H
#define PoissonInputSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "InputSpikingNeurons.h"

#include "../Helpers/RandomStateManager.h"

struct poisson_input_spiking_neuron_parameters_struct : input_spiking_neuron_parameters_struct {
	poisson_input_spiking_neuron_parameters_struct(): rate(50.0f) { input_spiking_neuron_parameters_struct(); }

	float rate;
};


class PoissonInputSpikingNeurons : public InputSpikingNeurons {
public:
	// Constructor/Destructor
	PoissonInputSpikingNeurons();
	~PoissonInputSpikingNeurons();

	float * rates;
	float * d_rates;

	virtual int AddGroup(neuron_parameters_struct * group_params);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);
	virtual void update_membrane_potentials(float timestep);

};


__global__ void poisson_update_membrane_potentials_kernel(curandState_t* d_states,
							float *d_rates,
							float *d_membrane_potentials_v,
							float timestep,
							float * d_thresholds_for_action_potential_spikes,
							size_t total_number_of_input_neurons,
							int current_stimulus_index);

#endif