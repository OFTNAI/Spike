#ifndef LIFSpikingNeurons_H
#define LIFSpikingNeurons_H

//CUDA #include <cuda.h>

#include "SpikingNeurons.hpp"


struct lif_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	lif_spiking_neuron_parameters_struct() : somatic_capcitance_Cm(0.0f), somatic_leakage_conductance_g0(0.0f), refractory_period_in_seconds(0.002f)  { spiking_neuron_parameters_struct(); }

	float somatic_capcitance_Cm;
	float somatic_leakage_conductance_g0;
	float refractory_period_in_seconds;

};

namespace Backend {
  class LIFSpikingNeurons : public SpikingNeurons {
  };
}

#include "Spike/Backend/Dummy/Neurons/LIFSpikingNeurons.hpp"

class LIFSpikingNeurons : public SpikingNeurons {
public:
  // Constructor/Destructor
  LIFSpikingNeurons();
  ~LIFSpikingNeurons();

  Backend::LIFSpikingNeurons* backend;
  
  float * membrane_time_constants_tau_m;
  float * membrane_resistances_R;

  float refractory_period_in_seconds;

  virtual void prepare_backend(Context* ctx);
  virtual int AddGroup(neuron_parameters_struct * group_params);

  // TODO:
  // virtual void update_membrane_potentials(float timestep,float current_time_in_seconds);

};

#endif
