#ifndef AdExSpikingNeurons_H
#define AdExSpikingNeurons_H

#include "SpikingNeurons.hpp"


struct AdEx_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	AdEx_spiking_neuron_parameters_struct() : membrane_capacitance_Cm(0.0f), membrane_leakage_conductance_g0(0.0f), absolute_refractory_period(0.002f), background_current(0.0f)  { spiking_neuron_parameters_struct(); }

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

class AdExSpikingNeurons; // forward definition

namespace Backend {
  class AdExSpikingNeurons : public virtual SpikingNeurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(AdExSpikingNeurons);
  };
}

class AdExSpikingNeurons : public SpikingNeurons {
public:
  ~AdExSpikingNeurons() override;
  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(AdExSpikingNeurons, SpikingNeurons);
  
  float * adaptation_values_w = nullptr;
  float * membrane_capacitances_Cm = nullptr;
  float * membrane_leakage_conductances_g0 = nullptr;
  float * leak_reversal_potentials_E_L = nullptr;
  float * slope_factors_Delta_T = nullptr;
  float * adaptation_coupling_coefficients_a = nullptr;
  float * adaptation_time_constants_tau_w = nullptr;
  float * adaptation_changes_b = nullptr;
  float absolute_refractory_period = 0;
  float background_current = 0;

  int AddGroup(neuron_parameters_struct * group_params) override;

private:
  std::shared_ptr<::Backend::AdExSpikingNeurons> _backend;
};

#endif
