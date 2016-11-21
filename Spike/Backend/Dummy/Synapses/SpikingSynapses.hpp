#pragma once

#include "Spike/Synapses/SpikingSynapses.hpp"

namespace Backend {
  namespace Dummy {
    class SpikingSynapses : public ::Backend::SpikingSynapses {
    public:
      // virtual void prepare() {}
      virtual void reset_state() {}

      void interact_spikes_with_synapses(::SpikingNeurons * neurons, ::SpikingNeurons * input_neurons, float current_time_in_seconds, float timestep) {
        // printf("TODO Dummy::SpikingSynapses::interact_spikes_with_synapses\n");
      }
    };
  } // namespace Dummy
} // namespace Backend

