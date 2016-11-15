#ifndef IzhikevichSpikingNeurons_H
#define IzhikevichSpikingNeurons_H

#include "SpikingNeurons.hpp"

struct izhikevich_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	izhikevich_spiking_neuron_parameters_struct(): parama(0.0f), paramb(0.0f), paramd(6.0f) { spiking_neuron_parameters_struct(); }

	float parama;
	float paramb;
	float paramd;
};

namespace Backend {
  class IzhikevichSpikingNeurons : public SpikingNeurons {
  };
}

class IzhikevichSpikingNeurons : public SpikingNeurons {
public:
  // Constructor/Destructor
  IzhikevichSpikingNeurons();
  ~IzhikevichSpikingNeurons();

  Backend::IzhikevichSpikingNeurons backend;
  
  float * param_a = NULL;
  float * param_b = NULL;
  float * param_d = NULL;

  virtual int AddGroup(neuron_parameters_struct * group_params);
  virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
  virtual void reset();
};


#endif
