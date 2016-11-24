#ifndef CURRENTSPIKINGSYNAPSES_H
#define CURRENTSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

class CurrentSpikingSynapses; // forward definition

namespace Backend {
  class CurrentSpikingSynapses : public SpikingSynapses {
  public:
    ADD_FRONTEND_GETTER(CurrentSpikingSynapses);
  };
}

#include "Spike/Backend/Dummy/Synapses/CurrentSpikingSynapses.hpp"

class CurrentSpikingSynapses : public SpikingSynapses {
public:
  ADD_BACKEND_GETTER(CurrentSpikingSynapses);
  
  virtual void AddGroup(int presynaptic_group_id, 
                        int postsynaptic_group_id, 
                        Neurons * neurons,
                        Neurons * input_neurons,
                        float timestep,
                        synapse_parameters_struct * synapse_params);


  virtual void calculate_postsynaptic_current_injection(SpikingNeurons * neurons, float current_time_in_seconds, float timestep);

  virtual void prepare_backend(Context* ctx = _global_ctx);
};

#endif
