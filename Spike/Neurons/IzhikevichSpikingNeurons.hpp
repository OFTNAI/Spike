#ifndef IzhikevichSpikingNeurons_H
#define IzhikevichSpikingNeurons_H

#include "SpikingNeurons.hpp"

struct izhikevich_spiking_neuron_parameters_struct : spiking_neuron_parameters_struct {
	izhikevich_spiking_neuron_parameters_struct(): parama(0.0f), paramb(0.0f), paramd(6.0f) { spiking_neuron_parameters_struct(); }

	float parama;
	float paramb;
	float paramd;
};

class IzhikevichSpikingNeurons; // forward definition

namespace Backend {
  class IzhikevichSpikingNeurons : public SpikingNeurons {
  };
}

#include "Spike/Backend/Dummy/Neurons/IzhikevichSpikingNeurons.hpp"

class IzhikevichSpikingNeurons : public SpikingNeurons {
public:
  // Constructor/Destructor
  IzhikevichSpikingNeurons();
  ~IzhikevichSpikingNeurons();

  ADD_BACKEND_GETTER(IzhikevichSpikingNeurons);
  
  float * param_a = nullptr;
  float * param_b = nullptr;
  float * param_d = nullptr;

  virtual int AddGroup(neuron_parameters_struct * group_params);
  virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
  virtual void reset_state();
};


#endif
