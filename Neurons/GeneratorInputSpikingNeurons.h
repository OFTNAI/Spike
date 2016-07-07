#ifndef GeneratorInputSpikingNeurons_H
#define GeneratorInputSpikingNeurons_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "InputSpikingNeurons.h"

struct generator_input_spiking_parameters_struct : input_spiking_neuron_parameters_struct {
	generator_input_spiking_parameters_struct() { input_spiking_neuron_parameters_struct(); }
};

class GeneratorInputSpikingNeurons : public InputSpikingNeurons {
public:
	// Constructor/Destructor
	GeneratorInputSpikingNeurons();
	~GeneratorInputSpikingNeurons();

	int** neuron_id_matrix_for_stimuli;
	float** spike_times_matrix_for_stimuli;
	int* number_of_spikes_in_stimuli;

	int* d_neuron_ids_for_stimulus;
	float* d_spike_times_for_stimulus;

	int length_of_longest_stimulus;


	virtual int AddGroup(neuron_parameters_struct * group_params);

	virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
	virtual void reset_neuron_activities();

	virtual void set_threads_per_block_and_blocks_per_grid(int threads);

	virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);
	virtual void update_membrane_potentials(float timestep);

	void AddStimulus(int spikenumber, int* ids, float* spiketimes);
};


__global__ void check_for_generator_spikes_kernel(int *d_neuron_ids_for_stimulus,
								float *d_spike_times_for_stimulus,
								float* d_last_spike_time_of_each_neuron,
								unsigned char* d_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								size_t number_of_spikes_in_stimulus,
								bool high_fidelity_spike_flag);

#endif