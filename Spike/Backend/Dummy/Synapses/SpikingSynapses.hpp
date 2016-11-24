#pragma once

#include "Spike/Synapses/SpikingSynapses.hpp"
#include "Synapses.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingSynapsesCommon : public virtual SynapsesCommon,
                                  public virtual ::Backend::SpikingSynapsesCommon {
    public:
      virtual void interact_spikes_with_synapses(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
        // printf("TODO Backend::Synapses::interact_spikes_with_synapses\n");
      }
    };

    class SpikingSynapses : public virtual SpikingSynapsesCommon,
                            public ::Backend::SpikingSynapses {
    public:
      // virtual void prepare() {}
      virtual void reset_state() {}
    };
  } // namespace Dummy
} // namespace Backend

