#ifndef LIFSpikingNeurons_H
#define LIFSpikingNeurons_H

//CUDA #include <cuda.h>

#include "SpikingNeurons.h"


struct lif_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	lif_spiking_neuron_parameters_struct() : somatic_capcitance_Cm(0.0f), somatic_leakage_conductance_g0(0.0f), refractory_period_in_seconds(0.002f)  { spiking_neuron_parameters_struct(); }

	float somatic_capcitance_Cm;
	float somatic_leakage_conductance_g0;
	float refractory_period_in_seconds;

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

	float refractory_period_in_seconds;


	virtual int AddGroup(neuron_parameters_struct * group_params);

	virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
	virtual void copy_constants_to_device();

	virtual void update_membrane_potentials(float timestep,float current_time_in_seconds);

};

/*CUDA
__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
                                               float * d_last_spike_time_of_each_neuron,
                                               float * d_membrane_resistances_R,
                                               float * d_membrane_time_constants_tau_m,
                                               float * d_resting_potentials,
                                               float* d_current_injections,
                                               float timestep,
                                               float current_time_in_seconds,
                                               float refactory_period_in_seconds,
                                               size_t total_number_of_neurons);
*/

#endif
