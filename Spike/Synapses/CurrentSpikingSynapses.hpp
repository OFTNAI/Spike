#ifndef CURRENTSPIKINGSYNAPSES_H
#define CURRENTSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

struct current_spiking_synapse_parameters_struct : spiking_synapse_parameters_struct {
	current_spiking_synapse_parameters_struct(): decay_term_tau(0.001f) { spiking_synapse_parameters_struct(); }

	float decay_term_tau;
};


class CurrentSpikingSynapses; // forward definition

namespace Backend {
  class CurrentSpikingSynapses : public virtual SpikingSynapses {
  public:
    SPIKE_ADD_BACKEND_FACTORY(CurrentSpikingSynapses);
  };
}

class CurrentSpikingSynapses : public SpikingSynapses {
public:
  CurrentSpikingSynapses() : SpikingSynapses() {};
  CurrentSpikingSynapses(int seedval) : SpikingSynapses(seedval) {};

  SPIKE_ADD_BACKEND_GETSET(CurrentSpikingSynapses, SpikingSynapses);
  void init_backend(Context* ctx = _global_ctx) override;
  
  vector<float> decay_terms_tau;
  void AddGroup(int presynaptic_group_id, 
                int postsynaptic_group_id, 
                Neurons * neurons,
                Neurons * input_neurons,
                float timestep,
                synapse_parameters_struct * synapse_params) override;
  void state_update( SpikingNeurons* input_neurons, SpikingNeurons* neurons, float current_time_in_seconds, float timestep) override;
private:
  std::shared_ptr<::Backend::CurrentSpikingSynapses> _backend;
};

#endif
