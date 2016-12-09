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
    virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep) = 0;
    virtual void update_membrane_potentials(float timestep, float current_time_in_seconds) = 0;
  };
}

#include "Spike/Backend/Dummy/Neurons/SpikingNeurons.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Neurons/SpikingNeurons.hpp"
#endif


class SpikingNeurons : public Neurons {
public:
  // Constructor/Destructor
  SpikingNeurons();
  ~SpikingNeurons();

  void prepare_backend(Context* ctx) override;
  ADD_BACKEND_GETTER(SpikingNeurons);
  
  // Variables
  int bitarray_length;
  int bitarray_maximum_axonal_delay_in_timesteps;
  bool high_fidelity_spike_flag;
	
  // Host Pointers
  float* after_spike_reset_membrane_potentials_c = nullptr;
  float* thresholds_for_action_potential_spikes = nullptr;
  unsigned char* bitarray_of_neuron_spikes = nullptr;

  float* last_spike_time_of_each_neuron = nullptr;
  float* membrane_potentials_v = nullptr;
  float* resting_potentials = nullptr;

  // Functions
  int AddGroup(neuron_parameters_struct * group_params) override;
  void reset_state() override;

  virtual void update_membrane_potentials(float timestep, float current_time_in_seconds) = 0;
  virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);
};

#endif
