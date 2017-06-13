#include "AdExSpikingNeurons.hpp"
#include <stdlib.h>
#include <stdio.h>

AdExSpikingNeurons::~AdExSpikingNeurons() {
  free(adaptation_values_w);
  free(membrane_capacitances_Cm);
  free(membrane_leakage_conductances_g0);
  free(leak_reversal_potentials_E_L);
  free(slope_factors_Delta_T);
  free(adaptation_coupling_coefficients_a);
  free(adaptation_time_constants_tau_w);
  free(adaptation_changes_b);
}

int AdExSpikingNeurons::AddGroup(neuron_parameters_struct * group_params){

  int new_group_id = SpikingNeurons::AddGroup(group_params);

  AdEx_spiking_neuron_parameters_struct * AdEx_spiking_group_params = (AdEx_spiking_neuron_parameters_struct*)group_params;

  adaptation_values_w = (float*)realloc(adaptation_values_w, total_number_of_neurons*sizeof(float));
  membrane_capacitances_Cm = (float*)realloc(membrane_capacitances_Cm, total_number_of_neurons*sizeof(float));
  membrane_leakage_conductances_g0 = (float*)realloc(membrane_leakage_conductances_g0, total_number_of_neurons*sizeof(float));
  leak_reversal_potentials_E_L = (float*)realloc(leak_reversal_potentials_E_L, total_number_of_neurons*sizeof(float));
  slope_factors_Delta_T = (float*)realloc(slope_factors_Delta_T, total_number_of_neurons*sizeof(float));
  adaptation_coupling_coefficients_a = (float*)realloc(adaptation_coupling_coefficients_a, total_number_of_neurons*sizeof(float));
  adaptation_time_constants_tau_w = (float*)realloc(adaptation_time_constants_tau_w, total_number_of_neurons*sizeof(float));
  adaptation_changes_b = (float*)realloc(adaptation_changes_b, total_number_of_neurons*sizeof(float));

  absolute_refractory_period = AdEx_spiking_group_params->absolute_refractory_period;
  background_current = AdEx_spiking_group_params->background_current;

  for (int i = total_number_of_neurons - number_of_neurons_in_new_group; i < total_number_of_neurons; i++) {
    adaptation_values_w[i] = 0.0f;
    membrane_capacitances_Cm[i] = AdEx_spiking_group_params->membrane_capacitance_Cm;
    membrane_leakage_conductances_g0[i] = AdEx_spiking_group_params->membrane_leakage_conductance_g0;
    leak_reversal_potentials_E_L[i] = AdEx_spiking_group_params->leak_reversal_potential_E_L;
    slope_factors_Delta_T[i] = AdEx_spiking_group_params->slope_factor_Delta_T;
    adaptation_coupling_coefficients_a[i] = AdEx_spiking_group_params->adaptation_coupling_coefficient_a;
    adaptation_time_constants_tau_w[i] = AdEx_spiking_group_params->adaptation_time_constant_tau_w;
    adaptation_changes_b[i] = AdEx_spiking_group_params->adaptation_change_b;
  }

  return new_group_id;
}

SPIKE_MAKE_INIT_BACKEND(AdExSpikingNeurons);
