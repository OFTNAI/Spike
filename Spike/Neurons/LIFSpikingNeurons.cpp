#include "LIFSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>


LIFSpikingNeurons::LIFSpikingNeurons() {
	membrane_time_constants_tau_m = nullptr;
	membrane_resistances_R = nullptr;
}


LIFSpikingNeurons::~LIFSpikingNeurons() {
	free(membrane_time_constants_tau_m);
	free(membrane_resistances_R);
}


int LIFSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){

	int new_group_id = SpikingNeurons::AddGroup(group_params);

	lif_spiking_neuron_parameters_struct * lif_spiking_group_params = (lif_spiking_neuron_parameters_struct*)group_params;

	refractory_period_in_seconds = lif_spiking_group_params->absolute_refractory_period;

	membrane_time_constants_tau_m = (float*)realloc(membrane_time_constants_tau_m, total_number_of_neurons*sizeof(float));
	membrane_resistances_R = (float*)realloc(membrane_resistances_R, total_number_of_neurons*sizeof(float));

	float membrane_time_constant_tau_m = lif_spiking_group_params->somatic_capacitance_Cm / lif_spiking_group_params->somatic_leakage_conductance_g0;
	float membrane_resistance_R = 1 / lif_spiking_group_params->somatic_leakage_conductance_g0;
	
	for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
		membrane_time_constants_tau_m[i] = membrane_time_constant_tau_m;
		membrane_resistances_R[i] = membrane_resistance_R;
	}

	background_current = lif_spiking_group_params->background_current;

	return new_group_id;
}

SPIKE_MAKE_INIT_BACKEND(LIFSpikingNeurons);

