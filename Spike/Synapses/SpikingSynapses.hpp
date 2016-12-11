#ifndef SPIKINGSYNAPSES_H
#define SPIKINGSYNAPSES_H

#include "Synapses.hpp"
#include "../Neurons/SpikingNeurons.hpp"

class SpikingSynapses; // forward definition

namespace Backend {
  class SpikingSynapses : public virtual Synapses {
  public:
    ADD_FRONTEND_GETTER(SpikingSynapses);
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
  ~SpikingSynapses();

  ADD_BACKEND_GETTER(SpikingSynapses);

  // Host Pointers
  int* delays = nullptr;
  bool* stdp = nullptr;

  // For spike array stuff
  int maximum_axonal_delay_in_timesteps = 0;

  // Synapse Functions
  virtual void AddGroup(int presynaptic_group_id, 
                        int postsynaptic_group_id, 
                        Neurons * neurons,
                        Neurons * input_neurons,
                        float timestep,
                        synapse_parameters_struct * synapse_params);

  virtual void init_backend(Context* ctx = _global_ctx);
  virtual void reset_state();
  virtual void increment_number_of_synapses(int increment);
  virtual void shuffle_synapses();

  virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds) = 0;
  virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) = 0;

  virtual void interact_spikes_with_synapses(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep);
};

#endif
