#ifndef SPIKINGSYNAPSES_H
#define SPIKINGSYNAPSES_H

#include "Synapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"

namespace Backend {
  class SpikingSynapses : public Synapses {
  public:
  };
}

#include "Spike/Backend/Dummy/Synapses/SpikingSynapses.hpp"

struct spiking_synapse_parameters_struct : synapse_parameters_struct {
	spiking_synapse_parameters_struct(): stdp_on(true) { synapse_parameters_struct(); }

	bool stdp_on;
	float delay_range[2];

};

class SpikingSynapses : public Synapses {
public:
  // Constructor/Destructor
  SpikingSynapses();
  ~SpikingSynapses();

  // Host Pointers
  int* delays = NULL;
  bool* stdp = NULL;

  // For spike array stuff
  int maximum_axonal_delay_in_timesteps = 0;

  // Synapse Functions
  virtual void AddGroup(int presynaptic_group_id, 
                        int postsynaptic_group_id, 
                        Neurons * neurons,
                        Neurons * input_neurons,
                        float timestep,
                        synapse_parameters_struct * synapse_params);

  virtual void reset_state();
  virtual void set_threads_per_block_and_blocks_per_grid(int threads);
  virtual void increment_number_of_synapses(int increment);
  virtual void shuffle_synapses();

  virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds);
  virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep);

  virtual void interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep);
};

#endif
