#ifndef SPIKINGSYNAPSES_H
#define SPIKINGSYNAPSES_H

#include "Synapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"

class SpikingSynapses; // forward definition

namespace Backend {
  class SpikingSynapses : public virtual Synapses {
  public:
    SPIKE_ADD_FRONTEND_GETTER(SpikingSynapses);
    virtual void calculate_postsynaptic_current_injection(::SpikingNeurons * neurons, float current_time_in_seconds, float timestep) = 0;
    virtual void interact_spikes_with_synapses(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) = 0;
  };
}

#include "Spike/Backend/Dummy/Synapses/SpikingSynapses.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Synapses/SpikingSynapses.hpp"
#endif

struct spiking_synapse_parameters_struct : synapse_parameters_struct {
  spiking_synapse_parameters_struct(): stdp_on(true) { synapse_parameters_struct(); }

  bool stdp_on;
  float delay_range[2];
};

class SpikingSynapses : public Synapses {
public:
  ~SpikingSynapses() override;

  SPIKE_ADD_BACKEND_GETSET(SpikingSynapses, Synapses);
  void init_backend(Context* ctx = _global_ctx) override;

  // Host Pointers
  int* delays = nullptr;
  bool* stdp = nullptr;

  // For spike array stuff
  int maximum_axonal_delay_in_timesteps = 0;

  // Synapse Functions
  void AddGroup(int presynaptic_group_id, 
                int postsynaptic_group_id, 
                Neurons * neurons,
                Neurons * input_neurons,
                float timestep,
                synapse_parameters_struct * synapse_params) override;

  void increment_number_of_synapses(int increment) override;
  void shuffle_synapses() override;

  virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds) = 0;
  virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) = 0;

  virtual void interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep);

private:
  std::shared_ptr<::Backend::SpikingSynapses> _backend;
};

#endif
