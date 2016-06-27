#ifndef IzhikevichSpikingNeurons_H
#define IzhikevichSpikingNeurons_H

#include <cuda.h>

#include "SpikingNeurons.h"


struct izhikevich_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	izhikevich_spiking_neuron_parameters_struct(): parama(0.0f), paramb(0.0f), paramd(6.0f) { spiking_neuron_parameters_struct(); }

	float parama;
	float paramb;
	float paramd;
};


class IzhikevichSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	IzhikevichSpikingNeurons();
	~IzhikevichSpikingNeurons();

	float * param_a;
	float * param_b;
	float * param_d;

	float * d_param_a;
	float * d_param_b;
	float * d_param_d;

	float * d_states_u;

	virtual int AddGroup(neuron_parameters_struct * group_params);
	virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
	virtual void reset_neurons();
	virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);
	virtual void update_membrane_potentials(float timestep);

};


// GPU Kernels
__global__ void reset_states_u_after_spikes_kernel(float *d_states_u,
								float * d_param_d,
								float* d_last_spike_time_of_each_neuron,
								float current_time_in_seconds,
								size_t total_number_of_neurons);

__global__ void izhikevich_update_membrane_potentials_kernel(float *d_membrane_potentials_v,
								float *d_states_u,
								float *d_param_a,
								float *d_param_b,
								float *d_current_injections,
								float timestep,
								size_t total_number_of_neurons);

#endif