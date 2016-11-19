#ifndef SpikingNeurons_H
#define SpikingNeurons_H

//CUDA #include <cuda.h>
#include <stdio.h>

#include "Neurons.hpp"


struct spiking_neuron_parameters_struct : neuron_parameters_struct {
	spiking_neuron_parameters_struct(): resting_potential_v0(-0.074f), threshold_for_action_potential_spike(0.03f), absolute_refractory_period(0.002f) { neuron_parameters_struct(); }

	float resting_potential_v0;
	float threshold_for_action_potential_spike;
	float absolute_refractory_period;
};

namespace Backend {
  class SpikingNeurons : public Neurons {
  public:
    virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep) = 0;
  };
}

#include "Spike/Backend/Dummy/Neurons/SpikingNeurons.hpp"

class SpikingNeurons : public Neurons {
public:
  // Constructor/Destructor
  SpikingNeurons();
  ~SpikingNeurons();

  Backend::SpikingNeurons* backend;
  
  // Variables
  int bitarray_length;
  int bitarray_maximum_axonal_delay_in_timesteps;
  bool high_fidelity_spike_flag;
	
  // Host Pointers
  float* after_spike_reset_membrane_potentials_c = NULL;
  float* thresholds_for_action_potential_spikes = NULL;
  unsigned char* bitarray_of_neuron_spikes = NULL;

  float* last_spike_time_of_each_neuron = NULL;
  float* membrane_potentials_v = NULL;
  float* resting_potentials = NULL;

  // Functions
  virtual void prepare_backend(Context* ctx);
  virtual int AddGroup(neuron_parameters_struct * group_params);
  virtual void update_membrane_potentials(float timestep, float current_time_in_seconds);
  virtual void check_for_neuron_spikes(float current_time_in_seconds, float timestep);
  virtual void reset_state();
};

#endif
