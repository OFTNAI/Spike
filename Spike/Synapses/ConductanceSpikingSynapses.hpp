#ifndef ConductanceSPIKINGSYNAPSES_H
#define ConductanceSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

struct conductance_spiking_synapse_parameters_struct : spiking_synapse_parameters_struct {
	conductance_spiking_synapse_parameters_struct(): reversal_potential_Vhat(0.0f), decay_term_tau_g(0.001f) { spiking_synapse_parameters_struct(); }

	float reversal_potential_Vhat;
	float decay_term_tau_g;
};

class ConductanceSpikingSynapses; // forward definition

namespace Backend {
  class ConductanceSpikingSynapses : public virtual SpikingSynapses {
  public:
    SPIKE_ADD_BACKEND_FACTORY(ConductanceSpikingSynapses);
  };
}

class ConductanceSpikingSynapses : public SpikingSynapses {

public:
  ConductanceSpikingSynapses() : SpikingSynapses() {};
  ConductanceSpikingSynapses(int seedval) : SpikingSynapses(seedval) {};
  ~ConductanceSpikingSynapses() override;

  SPIKE_ADD_BACKEND_GETSET(ConductanceSpikingSynapses, SpikingSynapses);
  void init_backend(Context* ctx = _global_ctx) override;

  float * synaptic_conductances_g = nullptr;
  vector<float> reversal_potentials_Vhat;
  vector<float> decay_terms_tau_g;

  // Synapse Functions
  void AddGroup(int presynaptic_group_id, 
                int postsynaptic_group_id, 
                Neurons * neurons,
                Neurons * input_neurons,
                float timestep,
                synapse_parameters_struct * synapse_params) override;

  void increment_number_of_synapses(int increment);
  void shuffle_synapses() override;

  void state_update(SpikingNeurons * neurons, SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) override;

private:
  std::shared_ptr<::Backend::ConductanceSpikingSynapses> _backend;
};

#endif
