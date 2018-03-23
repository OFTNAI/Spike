#ifndef LIFSpikingNeurons_H
#define LIFSpikingNeurons_H

#include "SpikingNeurons.hpp"


struct lif_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	lif_spiking_neuron_parameters_struct() : somatic_capacitance_Cm(0.0f), somatic_leakage_conductance_g0(0.0f), background_current(0.0f)  { spiking_neuron_parameters_struct(); }

	float somatic_capacitance_Cm;
	float somatic_leakage_conductance_g0;
  float background_current;

};

class LIFSpikingNeurons; // forward definition

namespace Backend {
  class LIFSpikingNeurons : public virtual SpikingNeurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(LIFSpikingNeurons);
  };
}

class LIFSpikingNeurons : public SpikingNeurons {
public:
  // Constructor/Destructor
  LIFSpikingNeurons();
  ~LIFSpikingNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(LIFSpikingNeurons, SpikingNeurons);
  
  float * membrane_time_constants_tau_m;
  float * membrane_resistances_R;
  float background_current;

  float refractory_period_in_seconds;

  int AddGroup(neuron_parameters_struct * group_params) override;

private:
  std::shared_ptr<::Backend::LIFSpikingNeurons> _backend;
};

#endif
