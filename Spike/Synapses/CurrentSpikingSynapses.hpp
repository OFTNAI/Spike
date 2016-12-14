#ifndef CURRENTSPIKINGSYNAPSES_H
#define CURRENTSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

class CurrentSpikingSynapses; // forward definition

namespace Backend {
  class CurrentSpikingSynapses : public virtual SpikingSynapses {
  public:
    ADD_FRONTEND_GETTER(CurrentSpikingSynapses);
  };
}

#include "Spike/Backend/Dummy/Synapses/CurrentSpikingSynapses.hpp"
#ifdef SPIKE_WITH_CUDA
#include "Spike/Backend/CUDA/Synapses/CurrentSpikingSynapses.hpp"
#endif

class CurrentSpikingSynapses : public SpikingSynapses {
public:
  ADD_BACKEND_GETSET(CurrentSpikingSynapses, SpikingSynapses);
  void init_backend(Context* ctx = _global_ctx) override;
  
  void AddGroup(int presynaptic_group_id, 
                int postsynaptic_group_id, 
                Neurons * neurons,
                Neurons * input_neurons,
                float timestep,
                synapse_parameters_struct * synapse_params) override;

  void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep) override;

private:
  ::Backend::CurrentSpikingSynapses* _backend = nullptr;
};

#endif
