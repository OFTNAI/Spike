#ifndef SpikingNeurons_H
#define SpikingNeurons_H

#include <cstdio>

#include "Neurons.hpp"


struct spiking_neuron_parameters_struct : neuron_parameters_struct {
	spiking_neuron_parameters_struct(): resting_potential_v0(-0.074f), threshold_for_action_potential_spike(0.03f), absolute_refractory_period(0.002f) { neuron_parameters_struct(); }

	float resting_potential_v0;
	float threshold_for_action_potential_spike;
	float absolute_refractory_period;
};

class SpikingNeurons; // forward definition

namespace Backend {
  class SpikingNeurons : public virtual Neurons {
  public:
    SPIKE_ADD_BACKEND_FACTORY(SpikingNeurons);
    virtual void state_update(float current_time_in_seconds, float timestep) = 0;
  };
}

class SpikingNeurons : public Neurons {
public:
  // Constructor/Destructor
  SpikingNeurons();
  ~SpikingNeurons() override;

  void init_backend(Context* ctx) override;
  SPIKE_ADD_BACKEND_GETSET(SpikingNeurons, Neurons);
  void prepare_backend_early() override;
  
  // Host Pointers
  float* after_spike_reset_membrane_potentials_c = nullptr;
  float* thresholds_for_action_potential_spikes = nullptr;
  unsigned char* bitarray_of_neuron_spikes = nullptr;

  // Functions
  int AddGroup(neuron_parameters_struct * group_params) override;

  virtual void state_update(float current_time_in_seconds, float timestep);

private:
  std::shared_ptr<::Backend::SpikingNeurons> _backend;
};

#endif
