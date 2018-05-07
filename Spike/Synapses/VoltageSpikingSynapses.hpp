#ifndef VoltageSPIKINGSYNAPSES_H
#define VoltageSPIKINGSYNAPSES_H

#include "SpikingSynapses.hpp"
#include "Spike/Neurons/SpikingNeurons.hpp"

struct voltage_spiking_synapse_parameters_struct : spiking_synapse_parameters_struct {
};

class VoltageSpikingSynapses; // forward definition

namespace Backend {
  class VoltageSpikingSynapses : public virtual SpikingSynapses {
  public:
    SPIKE_ADD_BACKEND_FACTORY(VoltageSpikingSynapses);
  };
}

class VoltageSpikingSynapses : public SpikingSynapses {

public:
  VoltageSpikingSynapses() : SpikingSynapses() {};
  VoltageSpikingSynapses(int seedval) : SpikingSynapses(seedval) {};
  ~VoltageSpikingSynapses() override;

  SPIKE_ADD_BACKEND_GETSET(VoltageSpikingSynapses, SpikingSynapses);
  void init_backend(Context* ctx = _global_ctx) override;

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
  std::shared_ptr<::Backend::VoltageSpikingSynapses> _backend;
};

#endif
