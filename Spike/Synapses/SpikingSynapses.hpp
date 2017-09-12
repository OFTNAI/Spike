#ifndef SPIKINGSYNAPSES_H
#define SPIKINGSYNAPSES_H

class SpikingSynapses; // forward definition

#include "Synapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"
#include <vector>

namespace Backend {
  class SpikingSynapses : public virtual Synapses {
  public:
    SPIKE_ADD_BACKEND_FACTORY(SpikingSynapses);
    virtual void copy_weights_to_host() = 0;
    virtual void state_update(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) = 0;
  };
}

struct spiking_synapse_parameters_struct : synapse_parameters_struct {
//  spiking_synapse_parameters_struct(): plasticity_ptr(nullptr) { synapse_parameters_struct(); }
  float delay_range[2];
};

class SpikingSynapses : public Synapses {
public:
  SpikingSynapses() : Synapses() {};
  SpikingSynapses(int seedval) : Synapses(seedval) {};
  ~SpikingSynapses() override;

  SPIKE_ADD_BACKEND_GETSET(SpikingSynapses, Synapses);
  void init_backend(Context* ctx = _global_ctx) override;

  // Host Pointers
  int* delays = nullptr;

  // For spike array stuff
  int maximum_axonal_delay_in_timesteps = 0;

  // Synapse Functions
  void AddGroup(int presynaptic_group_id, 
                int postsynaptic_group_id, 
                Neurons * neurons,
                Neurons * input_neurons,
                float timestep,
                synapse_parameters_struct * synapse_params) override;

  void increment_number_of_synapses(int increment);
  void shuffle_synapses() override;

  virtual void state_update(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep);

private:
  std::shared_ptr<::Backend::SpikingSynapses> _backend;
};

#endif
