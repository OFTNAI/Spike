#ifndef LIFSpikingNeurons_H
#define LIFSpikingNeurons_H

#include <cuda.h>

#include "SpikingNeurons.h"


struct lif_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	lif_spiking_neuron_parameters_struct() : somatic_capcitance_Cm(0.0f), somatic_leakage_conductance_g0(0.0f)  { spiking_neuron_parameters_struct(); }

	float somatic_capcitance_Cm;
	float somatic_leakage_conductance_g0;

};


class LIFSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	LIFSpikingNeurons();
	~LIFSpikingNeurons();

	float * membrane_time_constants_tau_m;
	float * membrane_resistances_R;

	float * d_membrane_time_constants_tau_m;
	float * d_membrane_resistances_R;


	virtual int AddGroup(neuron_parameters_struct * group_params);

	virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
	virtual void copy_constants_to_device();
	virtual void reset_neurons();

	virtual void update_membrane_potentials(float timestep);

};

__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
								float * d_membrane_resistances_R,
								float * d_membrane_time_constants_tau_m,
								float * d_resting_potentials,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons);

#endif