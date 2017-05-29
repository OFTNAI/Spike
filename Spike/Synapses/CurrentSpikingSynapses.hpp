#ifndef CURRENTSPIKINGSYNAPSES_H
#define CURRENTSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

class CurrentSpikingSynapses; // forward definition

namespace Backend {
  class CurrentSpikingSynapses : public virtual SpikingSynapses {
  public:
    SPIKE_ADD_BACKEND_FACTORY(CurrentSpikingSynapses);
  };
}

class CurrentSpikingSynapses : public SpikingSynapses {
public:
  SPIKE_ADD_BACKEND_GETSET(CurrentSpikingSynapses, SpikingSynapses);
  void init_backend(Context* ctx = _global_ctx) override;
  
  void AddGroup(int presynaptic_group_id, 
                int postsynaptic_group_id, 
                Neurons * neurons,
                Neurons * input_neurons,
                float timestep,
                synapse_parameters_struct * synapse_params) override;
  virtual void update_synaptic_conductances(float timestep, float current_time_in_seconds) override;
  void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) override;

private:
  std::shared_ptr<::Backend::CurrentSpikingSynapses> _backend;
};

#endif
