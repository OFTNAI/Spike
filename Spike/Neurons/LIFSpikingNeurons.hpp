#ifndef LIFSpikingNeurons_H
#define LIFSpikingNeurons_H

#include "SpikingNeurons.hpp"


struct lif_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	lif_spiking_neuron_parameters_struct() : somatic_capcitance_Cm(0.0f), somatic_leakage_conductance_g0(0.0f), refractory_period_in_seconds(0.002f)  { spiking_neuron_parameters_struct(); }

	float somatic_capcitance_Cm;
	float somatic_leakage_conductance_g0;
	float refractory_period_in_seconds;

};

class LIFSpikingNeurons; // forward definition

namespace Backend {
  class LIFSpikingNeurons : public virtual SpikingNeurons {
  public:
    ADD_FRONTEND_GETTER(LIFSpikingNeurons);
  };
}

#include "Spike/Backend/Dummy/Neurons/LIFSpikingNeurons.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Neurons/LIFSpikingNeurons.hpp"
#endif


class LIFSpikingNeurons : public SpikingNeurons {
public:
  // Constructor/Destructor
  LIFSpikingNeurons();
  ~LIFSpikingNeurons();

  void init_backend(Context* ctx) override;
  ADD_BACKEND_GETSET(LIFSpikingNeurons, SpikingNeurons);
  
  float * membrane_time_constants_tau_m;
  float * membrane_resistances_R;

  float refractory_period_in_seconds;

  int AddGroup(neuron_parameters_struct * group_params) override;
  void update_membrane_potentials(float timestep,float current_time_in_seconds) override;

private:
  ::Backend::LIFSpikingNeurons* _backend = nullptr;
};

#endif
