#ifndef PoissonSpikingNeurons_H
#define PoissonSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "SpikingNeurons.h"

#include "../Helpers/RandomStateManager.h"

struct poisson_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	poisson_spiking_neuron_parameters_struct(): rate(50.0f) { spiking_neuron_parameters_struct(); }

	float rate;
};


class PoissonSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	PoissonSpikingNeurons();
	~PoissonSpikingNeurons();

	float * rates;
	float * d_rates;

	int total_number_of_input_images;
	int current_stimulus_index;

	virtual int AddGroup(neuron_parameters_struct * group_params, int group_shape[2]);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();
	virtual void set_threads_per_block_and_blocks_per_grid(int threads);

	void generate_random_states();
	virtual void update_membrane_potentials(float timestep);

};


__global__ void poisson_update_membrane_potentials_kernal(curandState_t* d_states,
							float *d_rates,
							float *d_membrane_potentials_v,
							float timestep,
							float * d_thresholds_for_action_potential_spikes,
							size_t total_number_of_input_neurons,
							int current_stimulus_index);

#endif