#ifndef AdExSpikingNeurons_H
#define AdExSpikingNeurons_H

#include "SpikingNeurons.hpp"


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

};

class AdExSpikingNeurons; // forward definition

namespace Backend {
  class AdExSpikingNeurons : public virtual SpikingNeurons {
  public:
    ADD_FRONTEND_GETTER(AdExSpikingNeurons);
  };
}

#include "Spike/Backend/Dummy/Neurons/AdExSpikingNeurons.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Neurons/AdExSpikingNeurons.hpp"
#endif

class AdExSpikingNeurons : public SpikingNeurons {
public:
  // Constructor/Destructor
  AdExSpikingNeurons();
  ~AdExSpikingNeurons();

  ADD_BACKEND_GETTER(AdExSpikingNeurons);
  
  float * adaptation_values_w = nullptr;
  float * membrane_capacitances_Cm = nullptr;
  float * membrane_leakage_conductances_g0 = nullptr;
  float * leak_reversal_potentials_E_L = nullptr;
  float * slope_factors_Delta_T = nullptr;
  float * adaptation_coupling_coefficients_a = nullptr;
  float * adaptation_time_constants_tau_w = nullptr;
  float * adaptation_changes_b = nullptr;
  float absolute_refractory_period = 0;

  virtual int AddGroup(neuron_parameters_struct * group_params);
  virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
  virtual void reset_state();
};

#endif
