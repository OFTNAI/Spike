#ifndef AdExSpikingNeurons_H
#define AdExSpikingNeurons_H

#include <cuda.h>

#include "SpikingNeurons.h"


struct AdEx_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	AdEx_spiking_neuron_parameters_struct() : membrane_capacitance_Cm(0.0f), membrane_leakage_conductance_g0(0.0f), absolute_refractory_period(0.002f)  { spiking_neuron_parameters_struct(); }

	float membrane_capacitance_Cm;
	float membrane_leakage_conductance_g0;
	float leak_reversal_potential_E_L;
	float slope_factor_Delta_T;
	float adaptation_coupling_coefficient_a;
	float adaptation_time_constant_tau_w;
	float adaptation_change_b;
	float absolute_refractory_period;
	float background_current;

};


class AdExSpikingNeurons : public SpikingNeurons {
public:
	// Constructor/Destructor
	AdExSpikingNeurons();
	~AdExSpikingNeurons();

	float * adaptation_values_w;
	float * membrane_capacitances_Cm;
	float * membrane_leakage_conductances_g0;
	float * leak_reversal_potentials_E_L;
	float * slope_factors_Delta_T;
	float * adaptation_coupling_coefficients_a;
	float * adaptation_time_constants_tau_w;
	float * adaptation_changes_b;
	float absolute_refractory_period;
	float background_current;

	float * d_adaptation_values_w;
	float * d_membrane_capacitances_Cm;
	float * d_membrane_leakage_conductances_g0;
	float * d_leak_reversal_potentials_E_L;
	float * d_slope_factors_Delta_T;
	float * d_adaptation_coupling_coefficients_a;
	float * d_adaptation_time_constants_tau_w;
	float * d_adaptation_changes_b;

	virtual int AddGroup(neuron_parameters_struct * group_params);

	virtual void allocate_device_pointers(int maximum_axonal_delay_in_timesteps, bool high_fidelity_spike_storage);
	virtual void copy_constants_to_device();
	virtual void reset_neuron_activities();

	virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
	virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);

};


__global__ void check_for_neuron_spikes_kernel(float *d_membrane_potentials_v,
								float *d_adaptation_values_w,
								float * d_adaptation_changes_b,
								float *d_thresholds_for_action_potential_spikes,
								float *d_resting_potentials,
								float* d_last_spike_time_of_each_neuron,
								unsigned char* d_bitarray_of_neuron_spikes,
								int bitarray_length,
								int bitarray_maximum_axonal_delay_in_timesteps,
								float current_time_in_seconds,
								float timestep,
								size_t total_number_of_neurons,
								bool high_fidelity_spike_flag);

__global__ void AdEx_update_membrane_potentials(float *d_membrane_potentials_v,
								float * d_adaptation_values_w,
								float * d_adaptation_changes_b,
								float * d_membrane_capacitances_Cm,
								float * d_membrane_leakage_conductances_g0,
								float * d_leak_reversal_potentials_E_L,
								float * d_slope_factors_Delta_T,
								float * d_adaptation_coupling_coefficients_a,
								float * d_adaptation_time_constants_tau_w,
								float * d_current_injections,
								float * d_thresholds_for_action_potential_spikes,
								float * d_last_spike_time_of_each_neuron,
								float absolute_refractory_period,
								float background_current,
								float current_time_in_seconds,
								float timestep,
								size_t total_number_of_neurons);


#endif
