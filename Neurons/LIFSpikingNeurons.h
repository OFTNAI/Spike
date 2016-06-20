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

	float decay_term_tau_D;
	float model_parameter_alpha_D;


	virtual int AddGroup(neuron_parameters_struct * group_params);
	virtual void allocate_device_pointers();
	virtual void reset_neurons();
	virtual void update_postsynaptic_activities(float timestep, float current_time_in_seconds);

	virtual void update_membrane_potentials(float timestep);

};

__global__ void lif_update_membrane_potentials(float *d_membrane_potentials_v,
								float * d_membrane_resistances_R,
								float * d_membrane_time_constants_tau_m,
								float * d_resting_potentials,
								float* d_current_injections,
								float timestep,
								size_t total_number_of_neurons);

__global__ void lif_update_postsynaptic_activities_kernal(float timestep,
								size_t total_number_of_neurons,
								float * d_recent_postsynaptic_activities_D,
								float * d_last_spike_time_of_each_neuron,
								float current_time_in_seconds,
								float decay_term_tau_D,
								float model_parameter_alpha_D);

#endif